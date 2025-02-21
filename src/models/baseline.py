import torch
import torch.nn as nn

from embeddings.positional import PositionalEncoding
from embeddings.timestep import TimestepEmbedder
from embeddings.mlp import MLPConditionalEmbedding
from embeddings.transformer import TEConditionalEmbedding
from embeddings.stft import STFTEmbedding

class LSTMEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x

class BaseLineModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=8, channels=1, cond_model = "mlp", cond_features = None, device="cpu"):
        super(BaseLineModel, self).__init__()
        self.channels = channels
        self.context_size = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.lstm_embedding = LSTMEmbedding(self.input_size, self.hidden_size)
        
        self.positional_embedding = PositionalEncoding(self.hidden_size)
        self.emb_timestep = TimestepEmbedder(self.hidden_size, self.positional_embedding)
        
        self.multihead_attention = MultiHeadSelfAttention(self.hidden_size, num_heads)
        self.mlp = MLP(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.input_size)
        
        self.cond_features = cond_features
        self.cond_model = cond_model
        assert self.cond_model in {"mlp", "te", "stft"}, "Chosen conditioning model was not valid, the options are mlp, te and stft"
        if cond_model == "mlp":
            self.conditional_embedding = MLPConditionalEmbedding(self.input_size, self.hidden_size)
        if cond_model == "te":
            self.conditional_embedding = TEConditionalEmbedding(self.cond_features)
        if cond_model == "stft":
            self.conditional_embedding = STFTEmbedding(seq_len=self.input_size, device=self.device)
        
        self.fc1 = nn.Linear(16, self.hidden_size) #16 output after reshape

    def forward(self, x, t, cond_input=None):
        #print(f"input shape: {x.shape}")
        lstm_out = self.lstm_embedding(x)
        #print(f"LSTM embed shape: {lstm_out.shape}")
        
        time_embed = self.emb_timestep(t)
        #print(f"time embed shape: {time_embed.shape}")
        pos_emb = self.positional_embedding(time_embed)
        pos_emb = pos_emb.permute(1, 0, 2)
        time_embed = time_embed.permute(1, 0, 2)
        #print(f"pos embed shape: {pos_emb.shape}")
        combined = torch.cat((time_embed, pos_emb), dim=1)
        #print(f"combined shape: {combined.shape}")
        
        combined = torch.cat((combined, lstm_out), dim=1)
        #print(f"combined with lstm shape: {combined.shape}")
        
        attn_out = self.multihead_attention(combined)
        #print(f"attn_out shape: {attn_out.shape}")
        mlp_out = self.mlp(attn_out)
        #print(f"mlp_out shape: {mlp_out.shape}")
        
        if cond_input is not None:
            cond_emb = self.conditional_embedding(cond_input)
            #print(f"shape of stft embedding: {cond_emb.shape}")
            
            if self.cond_model == "te":
                cond_emb = cond_emb.permute(0,2,1)
            #print(f"cond_emb shape: {cond_emb.shape}")
            
            if self.cond_model == "stft":
                cond_emb = cond_emb.reshape(x.shape[0], self.input_size, -1) #x.shape[1] is batch size
                #print(f"shape of stft embedding after reshape: {cond_emb.shape}")
                cond_emb = self.fc1(cond_emb)
                #cond_emb = cond_emb.permute(1, 0, 2)
                #print(f"shape of stft embedding after fc1: {cond_emb.shape}")
            
            mlp_out = torch.cat((mlp_out, cond_emb), dim=1)
            #print(f"mlp_out shape: {mlp_out.shape}")
            #mlp_out = mlp_out.mean(dim=1, keepdim=True)
        mlp_out = mlp_out[:, :self.channels, :]
        #print(f"mlp output shape: {mlp_out.shape}")
        
        final_out = self.linear(mlp_out)
        #print(f"final output shape: {final_out.shape}")
        return final_out