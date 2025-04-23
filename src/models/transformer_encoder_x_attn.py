#Based on https://github.com/fahim-sikder/TransFusion
import torch
import torch.nn as nn

from embeddings.positional import PositionalEncoding
from embeddings.timestep import TimestepEmbedder
from embeddings.mlp import MLPConditionalEmbedding
from embeddings.transformer import TEConditionalEmbedding
from embeddings.stft import STFTEmbedding
from embeddings.fft import FFTEmbedding

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Cross attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer norm for query and key/value
        self.norm_query = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)
        
        # Optional feedforward network after attention
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_ff = nn.LayerNorm(latent_dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: The sequence to be enhanced with conditional info [seq_len, batch, features]
            key_value: The conditional embedding [cond_seq_len, batch, features]
        """
        # Apply normalization
        query_norm = self.norm_query(query)
        kv_norm = self.norm_kv(key_value)
        
        # Apply cross attention: query attends to key_value
        attn_output, _ = self.cross_attention(
            query=query_norm,
            key=kv_norm,
            value=kv_norm
        )
        
        # Add residual connection
        attn_output = query + attn_output
        
        # Apply feedforward network with residual connection
        ff_output = self.norm_ff(attn_output)
        ff_output = attn_output + self.ff(ff_output)
        
        return ff_output


class TransEncoder(nn.Module):
    def __init__(self, features, latent_dim=256, num_heads=8, num_layers=6, seq_len=15, cond_model="mlp", cond_features=None, dropout=0.1, activation='gelu', ff_size=1024, device="cpu"):
        
        super().__init__()
        self.model_name = "TransEncoder"
        self.channels = features
        self.self_condition = None
        self.context_size = None
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.seq_len = seq_len
                
        self.pos_enc = PositionalEncoding(self.latent_dim)
        self.emb_timestep = TimestepEmbedder(self.latent_dim, self.pos_enc)
        self.input_dim = nn.Linear(self.channels, self.latent_dim)
        self.output_dim = nn.Linear(self.latent_dim, self.channels)
        
        self.TransEncLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=False
        )

        self.TransEncodeR = nn.TransformerEncoder(
            self.TransEncLayer,
            num_layers=self.num_layers
        )
        
        # Add cross attention module
        self.cross_attention = CrossAttention(
            latent_dim=self.latent_dim,
            num_heads=self.num_heads // 2,  # Often cross-attention uses fewer heads
            dropout=self.dropout
        )
        
        self.cond_features = cond_features
        self.cond_model = cond_model
        assert self.cond_model in {"mlp", "te", "stft", "fft"}, "Chosen conditioning model was not valid, the options are mlp, te, fft and spectro"
        if cond_model == "mlp":
            self.conditional_embedding = MLPConditionalEmbedding(self.seq_len, self.latent_dim)
        if cond_model == "te":
            self.conditional_embedding = TEConditionalEmbedding(features=self.cond_features)
        if cond_model == "fft":
            self.conditional_embedding = FFTEmbedding(in_features=self.cond_features, hidden_size=self.latent_dim)
        if cond_model == "stft":
            self.conditional_embedding = STFTEmbedding(seq_len=self.channels, device=self.device)
           
        self.fc1 = nn.Linear(16, self.latent_dim)  # 16 is output of stft after reshape
   
    
    def forward(self, x, t, cond_input=None):
        # Initial processing of input
        if torch.isnan(x).any():
            print("NaN detected in input")
            x = torch.nan_to_num(x, nan=0.0)
        
        #print(f"init input shape: {x.shape}")
        x = torch.transpose(x, 1, 2)
        #print(f"input after transpose shape: {x.shape}")
        x = self.input_dim(x)
        #print(f"input after Lin shape: {x.shape}")
        x = torch.transpose(x, 0, 1)  # [seq_len, batch, features]
        #print(f"Lin after transpose shape: {x.shape}")
        
        
        # Time embedding
        embed = self.emb_timestep(t)
        #print(f"embed shape: {embed.shape}")
        time_added_data = embed + x
        #print(f"embed + x shape: {time_added_data.shape}")
        time_added_data = self.pos_enc(time_added_data)
        #print(f"pos enc shape: {time_added_data.shape}")
        
        # Transformer encoder
        trans_output = self.TransEncodeR(time_added_data)
        #print(f"trans output shape: {trans_output.shape}")
        
        # Handle conditional input
        if cond_input is not None:
            #print(f"init cond shape: {cond_input.shape}")
            cond_emb = self.conditional_embedding(cond_input)
            #print(f"output cond shape: {cond_emb.shape}")
            # Adjust dimensions based on conditioning model
            if self.cond_model == "mlp":
                cond_emb = cond_emb.permute(1, 0, 2)  # [seq_len, batch, features]
            
            if self.cond_model == "te" or self.cond_model == "fft":
                cond_emb = cond_emb.permute(2, 0, 1)  # [seq_len, batch, features]
            
            if self.cond_model == "stft":
                cond_emb = cond_emb.reshape(x.shape[1], self.seq_len, -1)  # x.shape[1] is batch size
                cond_emb = self.fc1(cond_emb)
                cond_emb = cond_emb.permute(1, 0, 2)  # [seq_len, batch, features]
            
            # Apply cross attention instead of mean pooling and addition
            trans_output = self.cross_attention(trans_output, cond_emb)
        
        # Final processing
        final_output = self.output_dim(trans_output)
        transposed_data = final_output.permute(1, 2, 0)
        
        return transposed_data