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

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.head_dim = hidden_dim // num_heads
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projection for query, key, value
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
        
        return self.output_proj(attention_output)
    
class MLP(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.1)  # Leaky ReLU as specified
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BaseLineModel(nn.Module):
    def __init__(self, seq_len, hidden_dim, cond_dim, num_heads=8, dropout=0.1, device= "cpu", cond_model="mlp", channels= 1):
        super(BaseLineModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.cond_model = cond_model
        self.cond_dim = cond_dim
        self.device = device
        
        self.channels = channels
        self.model_name = "BaseLine"
        self.context_size = None
    
        self.lstm = nn.LSTM(
            input_size=self.seq_len,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.pos_embedding = PositionalEncoding(hidden_dim)
        self.emb_timestep = TimestepEmbedder(self.hidden_dim, self.pos_embedding)
        
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.mlp = MLP(hidden_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        assert self.cond_model in {"mlp", "te", "fft", "stft"}, "Chosen conditioning model was not valid, the options are mlp, te, fft and stft"
        if cond_model == "mlp":
            self.cond_embedding = MLPConditionalEmbedding(self.seq_len, self.hidden_dim)
        if cond_model == "te":
            self.cond_embedding = TEConditionalEmbedding(self.cond_dim)
        if cond_model == "stft":
            self.cond_embedding = STFTEmbedding(seq_len=self.seq_len, device=device)
        if cond_model == "fft":
            self.cond_embedding = FFTEmbedding(in_features=self.cond_dim, hidden_size=self.hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, seq_len)
        
    def forward(self, noise_input, noise_level, conditional_info=None): 
        lstm_output, _ = self.lstm(noise_input)
        #print(f"lstm_output: {lstm_output.shape}")
        time_embed = self.emb_timestep(noise_level)
        #print(f"time_embed: {time_embed.shape}")
        pos_embed = self.pos_embedding(time_embed).permute(1, 0, 2) 
        #print(f"pos_embed: {pos_embed.shape}")
        combined = lstm_output + pos_embed
        #print(f"combined: {combined.shape}")
        attention_output = self.norm1(self.self_attention(combined))
        #print(f"attention_output: {attention_output.shape}")

        mlp_output = self.norm2(self.mlp(attention_output))
        #print(f"mlp_output: {mlp_output.shape}")
        if conditional_info is not None:
            cond_embedded = self.cond_embedding(conditional_info)  
            #print(f"cond_embedded: {cond_embedded.shape}")
            
            if self.cond_model == "te" or self.cond_model == "fft":
                cond_embedded = cond_embedded.permute(0,2,1)
            
            if self.cond_model == "stft":
                cond_embedded = cond_embedded.reshape(noise_input.shape[0], self.input_size, -1)
                cond_embedded = self.fc1(cond_embedded)
                
            #currently using global pooling maybe attn based combination or Feature-wise Linear Modulation (FiLM) would be good aswell?
            cond_pooled = cond_embedded.mean(dim=1, keepdim=True)
            cond_expanded = cond_pooled.expand(-1, mlp_output.shape[1], -1)
            #print(f"cond_expanded: {cond_expanded.shape}")
            
            combined_output = mlp_output + cond_expanded
            #print(f"combined_output: {combined_output.shape}")
        
        else:
            combined_output = mlp_output
            #print(f"combined_output: {combined_output.shape}")
        
        final_output = self.output_layer(combined_output) 
        #print(f"final_output: {final_output.shape}")
        
        return final_output