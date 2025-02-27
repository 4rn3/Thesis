import torch
import torch.nn as nn

class FFTEmbedding(nn.Module):
    def __init__(self, in_features, hidden_size, norm_output=True, learnable_real_weights=True, learnable_imag_weights=True, use_inverse_fft=True):
        super(FFTEmbedding, self).__init__()
        
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.norm_output = norm_output
        self.use_inverse_fft = use_inverse_fft
        
        if learnable_real_weights:
            self.real_projection = nn.Parameter(
                torch.Tensor(hidden_size, in_features)
            )
            nn.init.xavier_uniform_(self.real_projection)
        else:
            self.register_parameter('real_projection', None)
            
        if learnable_imag_weights:
            self.imag_projection = nn.Parameter(
                torch.Tensor(hidden_size, in_features)
            )
            nn.init.xavier_uniform_(self.imag_projection)
        else:
            self.register_parameter('imag_projection', None)
            
        self.real_norm = nn.LayerNorm([hidden_size])
        self.imag_norm = nn.LayerNorm([hidden_size])
            
    def forward(self, x):
        
        batch_size, features, seq_len = x.shape
        assert features == self.in_features, f"Expected {self.in_features} features, got {features}"
        
        x_fft = torch.fft.fft(x, dim=2) 
        
        x_real = x_fft.real  
        x_imag = x_fft.imag
        
        if self.real_projection is not None:
            x_real = x_real.transpose(1, 2)  
            x_real = torch.matmul(x_real, self.real_projection.t()) 
            x_real = x_real.transpose(1, 2)
            
        if self.imag_projection is not None:
            x_imag = x_imag.transpose(1, 2) 
            x_imag = torch.matmul(x_imag, self.imag_projection.t())  
            x_imag = x_imag.transpose(1, 2)
            
        if self.norm_output:
            x_real = x_real.transpose(1, 2)  
            x_imag = x_imag.transpose(1, 2)
            
            x_real = self.real_norm(x_real)
            x_imag = self.imag_norm(x_imag)
            
            x_real = x_real.transpose(1, 2)
            x_imag = x_imag.transpose(1, 2)
            
        if self.use_inverse_fft:
            x_complex = torch.complex(x_real, x_imag)
            
            output = torch.fft.ifft(x_complex, dim=2).real
        else:
            concat = torch.cat([x_real, x_imag], dim=1)
            output = concat[:, :self.hidden_size, :] 
            
        return output