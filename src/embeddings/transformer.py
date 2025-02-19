import torch
import torch.nn as nn

from embeddings.positional import PositionalEncoding

class TEConditionalEmbedding(nn.Module):
    def __init__(self, features=3, latent_dim=256, num_heads=8, num_layers = 6, dropout = 0.1, activation = 'gelu', ff_size = 1024):
        super().__init__()
        self.channels = features
        self.latent_dim  = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.pos_enc = PositionalEncoding(self.latent_dim)
        self.input_dim = nn.Linear(self.channels, self.latent_dim)
        self.output_dim = nn.Linear(self.latent_dim, self.latent_dim)
        self.TransEncLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                  nhead=self.num_heads,
                                                  dim_feedforward=self.ff_size,
                                                  dropout=self.dropout,
                                                  activation=self.activation)
        
        self.TransEncodeR = nn.TransformerEncoder(self.TransEncLayer,
                                                     num_layers=self.num_layers)
        
    def forward(self, cond_input):
        #print(f"Input shape: {cond_input.shape}")
        cond_input = torch.transpose(cond_input, 1, 2)
        #print(f"Transposed shape: {cond_input.shape}")
        lin_out = self.input_dim(cond_input)
        #print(f"Input dim shape after linear: {lin_out.shape}")
        lin_out = torch.transpose(lin_out, 0, 1)
        #print(f"Input dim shape after transpose: {lin_out.shape}")
        
        pos_encoded = self.pos_enc(lin_out)
        #print(f"shape of pos enc: {pos_encoded.shape}")
        
        trans_output = self.TransEncodeR(pos_encoded)
        #print(f"Transformer Encoded output shape: {trans_output.shape}")
        
        final_output = self.output_dim(trans_output)
        #print(f"Transformer Encoded after linear output shape: {final_output.shape}")
        
        transposed_data = final_output.permute(1, 2, 0)
        #print(f"Final output shape: {transposed_data.shape}")
        return transposed_data
        