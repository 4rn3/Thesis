import torch
import torch.nn as nn

from embeddings.positional import PositionalEncoding
from embeddings.timestep import TimestepEmbedder
from embeddings.mlp import MLPConditionalEmbedding
from embeddings.transformer import TEConditionalEmbedding
from embeddings.stft import STFTEmbedding
from embeddings.fft import FFTEmbedding

class TransEncoder(nn.Module):
    
    def __init__(self, features, latent_dim=256, num_heads=8, num_layers = 6, seq_len = 15 ,cond_model = "mlp", cond_features = None ,dropout = 0.1, activation = 'gelu', ff_size = 1024, device="cpu"):
        
        super().__init__()

        self.channels = features
        self.self_condition = None
        self.context_size = None
        self.latent_dim  = latent_dim
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
        self.TransEncLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                  nhead=self.num_heads,
                                                  dim_feedforward=self.ff_size,
                                                  dropout=self.dropout,
                                                  activation=self.activation)

        self.TransEncodeR = nn.TransformerEncoder(self.TransEncLayer,
                                                     num_layers=self.num_layers)
        
        self.cond_features = cond_features
        self.cond_model = cond_model
        assert self.cond_model in {"mlp", "te", "stft", "fft"}, "Chosen conditioning model was not valid, the options are mlp, te, fft and spectro"
        if cond_model == "mlp":
            self.conditional_embedding = MLPConditionalEmbedding(self.seq_len, self.latent_dim)
        if cond_model == "te":
            self.conditional_embedding = TEConditionalEmbedding(self.cond_features)
        if cond_model == "stft":
            self.conditional_embedding = STFTEmbedding(seq_len=self.channels, device=self.device)
        if cond_model == "fft":
            self.conditional_embedding = FFTEmbedding(in_features= self.cond_features, hidden_size=self.latent_dim)
           
        self.fc1 = nn.Linear(16, self.latent_dim) #16 is output of stft after reshape
        
    def forward(self, x, t, cond_input = None):
        #print(f"Input shape: {x.shape}")
        x = torch.transpose(x, 1, 2)
        #print(f"Transposed shape: {x.shape}")
        x = self.input_dim(x)
        #print(f"Input dim shape after linear: {x.shape}")
        x = torch.transpose(x, 0, 1)
        #print(f"Input dim shape after transpose: {x.shape}")
        embed = self.emb_timestep(t)
        #print(f"Time embedding shape: {embed.shape}")
        time_added_data = embed + x
        #print(f"Time added data shape: {time_added_data.shape}")
        time_added_data = self.pos_enc(time_added_data)
        #print(f"Time added data shape after pos enc: {time_added_data.shape}")
        trans_output = self.TransEncodeR(time_added_data)
        #print(f"Transformer Encoded output shape: {trans_output.shape}")
        
        if cond_input is not None:
            cond_emb = self.conditional_embedding(cond_input)
            #print(f"cond emb shape: {cond_emb.shape}")
            
            if self.cond_model == "mlp":
                cond_emb = cond_emb.permute(1,0,2)
            
            if self.cond_model == "te" or self.cond_model == "fft":
                cond_emb = cond_emb.permute(2,0,1)
            
            if self.cond_model == "stft":
                cond_emb = cond_emb.reshape(x.shape[1], self.seq_len, -1) #x.shape[1] is batch size
                cond_emb = self.fc1(cond_emb)
                cond_emb = cond_emb.permute(1, 0, 2)
                
            #print(f"Cond embed shape: {cond_emb.shape}")
            
            cond_pooled = cond_emb.mean(dim=0, keepdim=True)
            cond_expanded = cond_pooled.expand(trans_output.shape[0], -1, -1)
            trans_output = trans_output + cond_expanded
            #print(f"Combined transformer output shape: {trans_output.shape}")
        
        final_output = self.output_dim(trans_output)
        #print(f"Transformer Encoded after linear output shape: {final_output.shape}")
            
        transposed_data = final_output.permute(1, 2, 0)
        #print(f"Final output shape: {transposed_data.shape}")
        return transposed_data