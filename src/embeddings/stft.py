import torch
import torch.nn as nn
import torchaudio.transforms as T

class STFTEmbedding(nn.Module):
    def __init__(self, in_features, sequence_length, n_fft, hop_length, latent_dim):
        super().__init__()
        self.in_features = in_features
        self.sequence_length = sequence_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.latent_dim = latent_dim

        self.stft = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            center=True,
            pad_mode="reflect"
        )

        # Calculate the output shape of STFT to determine MLP input size
        with torch.no_grad():
            dummy_input_single_feature = torch.randn(1, self.sequence_length)
            stft_out_single_feature = self.stft(dummy_input_single_feature)
            self.num_freq_bins = stft_out_single_feature.shape[1]
            self.num_time_frames = stft_out_single_feature.shape[2]

        stft_feature_repr_dim = self.num_freq_bins * self.num_time_frames

        self.mlp = nn.Sequential(
            nn.Linear(stft_feature_repr_dim, latent_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, in_features, sequence_length)
        batch_size = x.shape[0]
        
        stft_results_per_feature = []
        for i in range(self.in_features):
            # current_feature_data shape: (batch_size, sequence_length)
            current_feature_data = x[:, i, :] 
            # stft_out shape: (batch_size, num_freq_bins, num_time_frames)
            stft_out = self.stft(current_feature_data) 
            stft_results_per_feature.append(stft_out)

        # x_stft shape: (batch_size, in_features, num_freq_bins, num_time_frames)
        x_stft = torch.stack(stft_results_per_feature, dim=1)
        
        x_stft_reshaped = x_stft.reshape(batch_size, self.in_features, -1)

        # MLP input: (batch_size, self.in_features, stft_feature_repr_dim)
        # MLP output: (batch_size, self.in_features, self.latent_dim)
        embedding = self.mlp(x_stft_reshaped)
        # print(f"Output embedding shape: {embedding.shape}")

        return embedding