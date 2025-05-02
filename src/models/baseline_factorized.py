import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Ensure noise_level is (B, 1)
        if noise_level.ndim == 1:
            noise_level = noise_level.unsqueeze(-1)
        # Ensure noise_level is float
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
    
class BaselineDenoisingNetworkFactorizedCond(nn.Module):
    """
    The Denoising Network based on Figure 3 (left) of the paper.
    Adapted for input shape [batch_size, features, sequence_length].
    Handles conditioning input y (B, C_per_step, S) by flattening.
    Uses FACTORIZED Linear layers for the initial conditioning MLP embedding.
    """
    def __init__(self, seq_len, features, cond_features_per_step, hidden_dim, num_heads=4, cond_intermediate_rank=None):
        """
        Initializes the BaselineDenoisingNetworkFactorizedCond.

        Args:
            seq_len (int): Length of the input sequence (S).
            features (int): Number of features in the input data (F).
            cond_features_per_step (int): Number of features per step in the conditioning data (C_per_step).
            hidden_dim (int): Hidden dimension for main processing path and conditioning MLP output (H).
            num_heads (int): Number of heads for Multi-Head Self-Attention.
            cond_intermediate_rank (int, optional): Intermediate rank for the factorized conditioning MLP.
                                                    Defaults to hidden_dim if None.
        """
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.cond_features_per_step = cond_features_per_step
        self.cond_features_total = cond_features_per_step * seq_len # Calculate total flattened features

        # --- Main Path Modules ---
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

        # --- Conditioning Path Module (Factorized MLP) ---
        # Use factorized MLP: Linear -> Linear -> GELU -> Linear
        # Input: (B, C_total), Output: (B, H)
        if cond_intermediate_rank is None:
            cond_intermediate_rank = hidden_dim # Default to hidden_dim

        print(f"Using factorized conditioning MLP with intermediate rank: {cond_intermediate_rank}")

        self.cond_embedding = nn.Sequential(
            nn.Linear(self.cond_features_total, cond_intermediate_rank), # Factor 1: Maps C*S -> rank
            # Optional activation/norm could go here
            nn.Linear(cond_intermediate_rank, hidden_dim * 2), # Factor 2: Maps rank -> H*2
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim) # Maps H*2 -> H
        )


        # --- Final Layer ---
        self.final_layer = nn.Linear(hidden_dim, features)


    def forward(self, x_n, sqrt_alpha_bar, y):
        """
        Forward pass using factorized MLP conditioning.

        Args:
            x_n (torch.Tensor): Noisy input data, shape (B, F, S).
            sqrt_alpha_bar (torch.Tensor): Noise level for each batch item, shape (B,) or (B, 1).
            y (torch.Tensor): Conditioning information, shape (B, C_per_step, S).

        Returns:
            torch.Tensor: Predicted noise epsilon_theta, shape (B, F, S).
        """
        batch_size = x_n.shape[0]
        seq_len = x_n.shape[2] # S

        # --- Prepare Inputs ---
        x_n_permuted = x_n.permute(0, 2, 1) # (B, F, S) -> (B, S, F)

        # Ensure y has the correct shape (B, C, S)
        if y.ndim != 3 or y.shape[0] != batch_size or y.shape[1] != self.cond_features_per_step or y.shape[2] != seq_len:
                 raise ValueError(f"Unexpected shape for conditioning input y: {y.shape}. Expected (B, C_per_step, S) = ({batch_size}, {self.cond_features_per_step}, {seq_len})")

        # Flatten y for the MLP: (B, C, S) -> (B, C*S)
        y_flat = y.flatten(start_dim=1)


        # --- Process Main Path ---
        lstm_out, _ = self.lstm_embedding(x_n_permuted) # (B, S, H)

        # Process sqrt_alpha_bar (noise level)
        # Ensure sqrt_alpha_bar is (B, 1) for linear layer or embedding
        if sqrt_alpha_bar.ndim == 1:
             sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(1) # (B,) -> (B, 1)
        # Make sure PositionalEmbedding handles (B, 1) or modify call
        # pos_emb = self.pos_embedding(sqrt_alpha_bar) # Using actual PositionalEmbedding
        pos_emb = self.pos_embedding(sqrt_alpha_bar.float()) # Using placeholder Linear layer

        pos_emb_unsqueezed = pos_emb.unsqueeze(1) # (B, 1, H)
        combined_emb = lstm_out + pos_emb_unsqueezed # Add positional embedding

        attn_output, _ = self.attention(combined_emb, combined_emb, combined_emb)
        attn_output_res = attn_output + combined_emb
        attn_output_norm = self.attn_norm(attn_output_res) # (B, S, H)

        mlp_output = self.mlp(attn_output_norm)
        mlp_output_res = mlp_output + attn_output_norm
        mlp_output_norm = self.mlp_norm(mlp_output_res) # (B, S, H) - Main path output

        # --- Process Conditioning Path using Factorized MLP ---
        # Input y_flat shape is (B, C*S)
        cond_global_emb = self.cond_embedding(y_flat) # Output shape: (B, H)

        # --- Combine Main Path and Conditioning ---
        # Add the global conditioning vector (broadcasted across sequence length S)
        final_repr = mlp_output_norm + cond_global_emb.unsqueeze(1) # (B, S, H) + (B, 1, H) -> (B, S, H)

        # --- Final Layer ---
        predicted_noise_permuted = self.final_layer(final_repr) # (B, S, F)
        predicted_noise = predicted_noise_permuted.permute(0, 2, 1) # (B, F, S)

        return predicted_noise