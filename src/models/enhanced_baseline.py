#Based on Generating Synthetic Net Load Data with Physics-informed Diffusion Model by Zhang et al
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from embeddings.positional import PositionalEncoding
from embeddings.timestep import TimestepEmbedder
from embeddings.mlp import MLPConditionalEmbedding
from embeddings.transformer import TEConditionalEmbedding
from embeddings.stft import STFTEmbedding
from embeddings.fft import FFTEmbedding

class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, use_gating=True, cross_attention=False):
        super(EnhancedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.use_gating = use_gating
        self.cross_attention = cross_attention
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.head_dim = hidden_dim // num_heads
        
        if cross_attention:
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        else:
            self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
            
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        if use_gating:
            self.gate = nn.Linear(hidden_dim, hidden_dim)
            self.gate_activation = nn.Sigmoid()
        
    def forward(self, x, context=None):
        residual = x
        x = self.norm(x)
        
        batch_size, seq_len, _ = x.shape
        
        if self.cross_attention and context is not None:
            q = self.q_proj(x)
            kv = self.kv_proj(context)
            
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
            
            context_len = context.shape[1]
            kv = kv.reshape(batch_size, context_len, 2, self.num_heads, self.head_dim)
            kv = kv.permute(2, 0, 3, 1, 4)  # [2, batch, heads, context_len, head_dim]
            
            k, v = kv[0], kv[1]
        else:
            qkv = self.qkv_proj(x)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            
            q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, self.hidden_dim)
        
        output = self.output_proj(attention_output)
        
        if self.use_gating:
            gate_value = self.gate_activation(self.gate(x))
            output = output * gate_value
        
        return output + residual

class EnhancedBaseLineModel(nn.Module):
    def __init__(self, seq_len, hidden_dim, cond_dim, num_heads=8, num_layers=2, 
                 dropout=0.1, device="cpu", cond_model="mlp", channels=1):
        super(EnhancedBaseLineModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.cond_model = cond_model
        self.cond_dim = cond_dim
        self.device = device
        self.num_layers = num_layers
        
        self.channels = channels
        self.model_name = "BaseLine"
        self.context_size = None
    
        self.lstm = nn.LSTM(
            input_size=self.seq_len,
            hidden_size=hidden_dim,
            num_layers=2,  # Increased to 2 layers
            batch_first=True,
            bidirectional=True,  # Using bidirectional for better context capture
            dropout=dropout if num_layers > 1 else 0
        )
        # Project bidirectional output back to hidden_dim
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.pos_embedding = PositionalEncoding(hidden_dim)
        self.emb_timestep = TimestepEmbedder(self.hidden_dim, self.pos_embedding)
        
        # more attn 
        self.attention_layers = nn.ModuleList([
            EnhancedMultiHeadAttention(hidden_dim, num_heads, dropout, cross_attention=False)
            for _ in range(num_layers)
        ])
        
        # MLP with residual connections
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_final = nn.LayerNorm(hidden_dim)
        
        # Conditional embedding
        assert self.cond_model in {"mlp", "te", "fft", "stft"}, "Chosen conditioning model was not valid"
        if cond_model == "mlp":
            self.cond_embedding = MLPConditionalEmbedding(self.seq_len, self.hidden_dim)
        elif cond_model == "te":
            self.cond_embedding = TEConditionalEmbedding(self.cond_dim)
        elif cond_model == "stft":
            self.cond_embedding = STFTEmbedding(seq_len=self.seq_len, device=device)
        elif cond_model == "fft":
            self.cond_embedding = FFTEmbedding(in_features=self.cond_dim, hidden_size=self.hidden_dim)
        
        # Cross-attention for better conditioning
        self.cross_attention = EnhancedMultiHeadAttention(hidden_dim, num_heads, dropout, cross_attention=True)
        
        self.output_layer = nn.Linear(hidden_dim, seq_len)
        
    def forward(self, noise_input, noise_level, conditional_info=None): 
        
        lstm_output, _ = self.lstm(noise_input)
        if lstm_output.shape[-1] != self.hidden_dim:
            lstm_output = self.lstm_proj(lstm_output)
        
        time_embed = self.emb_timestep(noise_level)
        pos_embed = self.pos_embedding(time_embed).permute(1, 0, 2) 
        
        x = lstm_output + pos_embed
        
        for attn_layer in self.attention_layers:
            x = attn_layer(x)
        
        x = x + self.mlp(x)
        x = self.norm_final(x)
        
        if conditional_info is not None:
            cond_embedded = self.cond_embedding(conditional_info)  
            
            if self.cond_model == "te" or self.cond_model == "fft":
                cond_embedded = cond_embedded.permute(0, 2, 1)
            
            if self.cond_model == "stft":
                cond_embedded = cond_embedded.reshape(noise_input.shape[0], self.seq_len, -1)
            
            # Use cross-attention instead of simple mean pooling
            x = self.cross_attention(x, cond_embedded)
        
        return self.output_layer(x)