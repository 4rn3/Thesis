import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        if noise_level.ndim == 1:
            noise_level = noise_level.unsqueeze(-1)
        noise_level = noise_level.float()

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=noise_level.device) * -embeddings)
        # Shape: (1, half_dim)
        embeddings = noise_level * embeddings.unsqueeze(0)
        # Shape: (B, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Shape: (B, dim) if dim is even, else (B, dim-1)
        # Handle odd dim
        if self.dim % 2 == 1:
           embeddings = F.pad(embeddings, (0,1))

        return embeddings
    
class BaselineDenoisingNetwork(nn.Module):
    def __init__(self, seq_len, features, cond_features_per_step, hidden_dim, num_heads=4):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.cond_features_per_step = cond_features_per_step
        self.cond_features_total = cond_features_per_step * seq_len # Calculate total flattened features

        self.lstm_embedding = nn.LSTM(
            input_size=features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.pos_embedding = PositionalEmbedding(hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)

        # Input: (B, C_total), Output: (B, H)        
        self.cond_embedding = nn.Sequential(
            nn.Linear(self.cond_features_total, hidden_dim * 2), # Maps C*S -> H*2
            nn.GELU(), # Or nn.LeakyReLU()
            nn.Linear(hidden_dim * 2, hidden_dim) # Maps H*2 -> H
        )


        self.final_layer = nn.Linear(hidden_dim, features)


    def forward(self, x_n, sqrt_alpha_bar, y):

        # print(f"--- Denoising Network Forward (Flattened MLP Conditioning) ---")
        # print(f"Initial x_n shape: {x_n.shape}")
        # print(f"Initial sqrt_alpha_bar shape: {sqrt_alpha_bar.shape}")
        # print(f"Initial y shape: {y.shape}")
        batch_size = x_n.shape[0]
        seq_len = x_n.shape[2] # S

        x_n_permuted = x_n.permute(0, 2, 1) # (B, F, S) -> (B, S, F)
        # print(f"x_n_permuted shape (for LSTM): {x_n_permuted.shape}")

        # Ensure y has the correct shape (B, C, S)
        if y.ndim != 3 or y.shape[0] != batch_size or y.shape[1] != self.cond_features_per_step or y.shape[2] != seq_len:
             raise ValueError(f"Unexpected shape for conditioning input y: {y.shape}. Expected (B, C_per_step, S) = ({batch_size}, {self.cond_features_per_step}, {seq_len})")

        # Flatten y for the MLP: (B, C, S) -> (B, C*S)
        y_flat = y.flatten(start_dim=1)
        # print(f"y_flat shape (for Cond MLP): {y_flat.shape}")


        lstm_out, _ = self.lstm_embedding(x_n_permuted) # (B, S, H)
        # print(f"lstm_out shape: {lstm_out.shape}")

        pos_emb = self.pos_embedding(sqrt_alpha_bar) # (B, H)
        # print(f"pos_emb shape: {pos_emb.shape}")
        pos_emb_unsqueezed = pos_emb.unsqueeze(1) # (B, 1, H)
        # print(f"pos_emb_unsqueezed shape: {pos_emb_unsqueezed.shape}")
        combined_emb = lstm_out + pos_emb_unsqueezed # Add positional embedding
        # print(f"combined_emb shape (LSTM + PosEmb): {combined_emb.shape}")

        attn_output, _ = self.attention(combined_emb, combined_emb, combined_emb)
        # print(f"attn_output shape (raw): {attn_output.shape}")
        attn_output_res = attn_output + combined_emb
        # print(f"attn_output_res shape (before norm): {attn_output_res.shape}")
        attn_output_norm = self.attn_norm(attn_output_res) # (B, S, H)
        # print(f"attn_output_norm shape (after norm): {attn_output_norm.shape}")

        mlp_output = self.mlp(attn_output_norm)
        # print(f"mlp_output shape (raw): {mlp_output.shape}")
        mlp_output_res = mlp_output + attn_output_norm
        # print(f"mlp_output_res shape (before norm): {mlp_output_res.shape}")
        mlp_output_norm = self.mlp_norm(mlp_output_res) # (B, S, H) - Main path output
        # print(f"mlp_output_norm shape (main path final): {mlp_output_norm.shape}")

        # Input y_flat shape is (B, C*S)
        # Apply the MLP defined in __init__
        cond_global_emb = self.cond_embedding(y_flat) # Output shape: (B, H)
        # print(f"cond_global_emb shape: {cond_global_emb.shape}")

        # Add the global conditioning vector (broadcasted across sequence length S)
        # Unsqueeze to (B, 1, H) for broadcasting
        final_repr = mlp_output_norm + cond_global_emb.unsqueeze(1) # (B, S, H) + (B, 1, H) -> (B, S, H)
        # print(f"final_repr shape (Main + Global Cond): {final_repr.shape}")

        predicted_noise_permuted = self.final_layer(final_repr) # (B, S, F)
        # print(f"predicted_noise_permuted shape (before final permute): {predicted_noise_permuted.shape}")

        predicted_noise = predicted_noise_permuted.permute(0, 2, 1) # (B, F, S)
        # print(f"predicted_noise shape (final output): {predicted_noise.shape}")
        # print(f"--- Denoising Network Forward End ---")

        return predicted_noise