import torch.nn as nn

class MLPConditionalEmbedding(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()
        self.input_size = seq_len
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, x):
        #print(f"Shape x: {x.shape}")
        x = self.fc1(x)
        #print(f"Shape after fc1: {x.shape}")
        x = self.leaky_relu(x)
        x = self.fc2(x)
        #print(f"Shape after fc2: {x.shape}")
        return x