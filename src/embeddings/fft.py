import torch
import torch.nn as nn
import torch.fft

class FFTEmbedding(nn.Module):
    def __init__(self, seq_len, hidden_size, use_phase=False):
        super().__init__()
        self.original_seq_len = seq_len
        self.hidden_size = hidden_size
        self.use_phase = use_phase

        num_fft_components = self.original_seq_len // 2 + 1

        if self.use_phase:
            self.mlp_input_size = 2 * num_fft_components
        else:
            self.mlp_input_size = num_fft_components
        
        self.fc1 = nn.Linear(self.mlp_input_size, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, x):

        if not x.is_floating_point():
            x = x.float()

        # x shape: (batch_size, self.original_seq_len)
        # print(f"Input x shape: {x.shape}")
        
        x_fft = torch.fft.rfft(x, n=self.original_seq_len, dim=-1)
        # print(f"Shape after rfft: {x_fft.shape}, dtype: {x_fft.dtype}")

        if self.use_phase:
            # real and imaginary parts
            x_real = x_fft.real
            x_imag = x_fft.imag
            mlp_input = torch.cat((x_real, x_imag), dim=-1)
            # Expected shape for mlp_input: (batch_size, 2 * (self.original_seq_len // 2 + 1))
        else:
            # magnitudes
            mlp_input = x_fft.abs()
            # Expected shape for mlp_input: (batch_size, self.original_seq_len // 2 + 1)
        
        # print(f"Shape of mlp_input: {mlp_input.shape}")

        out = self.fc1(mlp_input)
        # print(f"Shape after fc1: {out.shape}")
        out = self.leaky_relu(out)
        out = self.fc2(out)
        # print(f"Shape after fc2 (output): {out.shape}")
        return out