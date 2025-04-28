import torch.nn as nn

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        time_embedded = self.time_embed(self.sequence_pos_encoder.pe[timesteps])
        if time_embedded.dim() == 2:
            return time_embedded.unsqueeze(0)
        else:
            return time_embedded.permute(1, 0, 2)