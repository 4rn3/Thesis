import torch
import torch.nn as nn

class STFTEmbedding(nn.Module):
    def __init__(self, seq_len, hop_length=None, win_length=15, latent_dim=256, device="cpu"):
        super(STFTEmbedding, self).__init__()
        self.n_fft = seq_len
        self.device = device
        self.hop_length = hop_length if hop_length is not None else self.n_fft // 4
        self.win_length = win_length if win_length is not None else self.n_fft
        self.window = torch.hann_window(self.win_length)
        self.window = self.window.to(self.device)
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(240, 15*self.latent_dim) #240 = shape of -1 at reshape
        
    def forward(self, x):  
        # x should have shape (batch, features, seq_len)
        batch_size, features, seq_len = x.shape
        
        
        stft_results = []
        for i in range(features):
            stft_result = torch.stft(x[:, i, :], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=True)
            stft_results.append(stft_result)

        # Stack the STFT results along the feature dimension
        stft_results = torch.stack(stft_results, dim=1)

        # Return the magnitude and phase as separate channels
        magnitude = stft_results.abs()
        phase = stft_results.angle()

        # Concatenate magnitude and phase along the feature dimension
        stft_embedding = torch.cat((magnitude, phase), dim=1)
        # stft_embedding = stft_embedding.reshape(32, -1)
        
        # out = self.fc1(stft_embedding)
        # out = out.reshape(32, 15, -1)
        #print(f"STFT Embedding output shape: {out.shape}")
        
        return stft_embedding