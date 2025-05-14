import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FeatureEmbeddingMLP(nn.Module): 
    def __init__(self, feature_embedding_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_embedding_dim, output_dim),
            nn.ReLU(), # or SiLU
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, emb):
        return self.mlp(emb)

class ConditionalProcessor(nn.Module):
    def __init__(self, cond_in_channels, sequence_length, cond_embedding_dim, mlp_hidden_factor=4):
        super().__init__()
        # Calculate the input dimension after flattening
        input_dim = cond_in_channels * sequence_length
        hidden_dim = cond_embedding_dim * mlp_hidden_factor
        
        self.mlp = nn.Sequential(
            nn.Flatten(), # Flattens input from (B, C_cond, S) to (B, C_cond * S)
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cond_embedding_dim)
        )
        
    def forward(self, c):
        # c shape: (batch_size, cond_in_channels, sequence_length)
        # Output shape: (batch_size, cond_embedding_dim)
        return self.mlp(c)

################ U-Net start ################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(mid_channels) # or LayerNorm
        self.act1 = nn.ReLU() # or SiLU, GELU
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(out_channels) # or LayerNorm
        self.act2 = nn.ReLU() # or SiLU, GELU

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x
    
######## Downblock ########
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, cond_emb_dim=None): 
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool1d(2)
        
        if time_emb_dim is not None:
            self.time_mlp = FeatureEmbeddingMLP(time_emb_dim, out_channels)
        else:
            self.time_mlp = None
            
        if cond_emb_dim is not None: 
            self.cond_mlp = FeatureEmbeddingMLP(cond_emb_dim, out_channels)
        else:
            self.cond_mlp = None

    def forward(self, x, t_emb=None, c_emb=None): 
        x = self.conv_block(x)
        
        if self.time_mlp is not None and t_emb is not None:
            time_info = self.time_mlp(t_emb).unsqueeze(-1) 
            x = x + time_info
            
        if self.cond_mlp is not None and c_emb is not None:
            cond_info = self.cond_mlp(c_emb).unsqueeze(-1) 
            x = x + cond_info
            
        skip = x 
        x = self.pool(x)
        return x, skip
    
######## Upblock ########
class UpBlock(nn.Module):
    def __init__(self, in_channels_up, in_channels_skip, out_channels, time_emb_dim=None, cond_emb_dim=None): 
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels_up, in_channels_up, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels_up + in_channels_skip, out_channels)
        
        if time_emb_dim is not None:
            self.time_mlp = FeatureEmbeddingMLP(time_emb_dim, out_channels)
        else:
            self.time_mlp = None

        if cond_emb_dim is not None: 
            self.cond_mlp = FeatureEmbeddingMLP(cond_emb_dim, out_channels)
        else:
            self.cond_mlp = None

    def forward(self, x, skip_connection, t_emb=None, c_emb=None): 
        x = self.up(x) 

        if x.shape[-1] != skip_connection.shape[-1]:
            diff = skip_connection.shape[-1] - x.shape[-1]
            x = F.pad(x, (diff // 2, diff - diff // 2))

        x = torch.cat([skip_connection, x], dim=1) 
        x = self.conv_block(x)
        
        if self.time_mlp is not None and t_emb is not None:
            time_info = self.time_mlp(t_emb).unsqueeze(-1)
            x = x + time_info
            
        if self.cond_mlp is not None and c_emb is not None:
            cond_info = self.cond_mlp(c_emb).unsqueeze(-1)
            x = x + cond_info
            
        return x
    
######## U-Net ########
class Unet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 time_embedding_dim=32, 
                 cond_in_channels=None, 
                 cond_sequence_length=None,
                 cond_embedding_dim=None, 
                 base_channels=32,
                 channel_mults=(1, 2, 4)):
        super().__init__()

        # Time projection layers
        self.time_projection = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )

        # Conditional embedding
        self.cond_embedding_dim = None 
        if cond_in_channels is not None and cond_embedding_dim is not None:
            if cond_sequence_length is None: 
                raise ValueError(
                    "cond_sequence_length must be provided if cond_in_channels and "
                    "cond_embedding_dim are set, as the ConditionalProcessor is now an MLP."
                )
            self.conditional_processor = ConditionalProcessor(
                cond_in_channels=cond_in_channels,
                sequence_length=cond_sequence_length,
                cond_embedding_dim=cond_embedding_dim
            )
            self.cond_embedding_dim = cond_embedding_dim # Store for passing to blocks
        else:
            self.conditional_processor = None

        self.initial_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        current_channels = base_channels

        # Down blocks
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.downs.append(DownBlock(current_channels, out_ch, 
                                        time_emb_dim=time_embedding_dim,
                                        cond_emb_dim=self.cond_embedding_dim)) 
            current_channels = out_ch

        # Bottleneck
        bottleneck_intermediate_channels = current_channels * 2
        self.bottleneck_conv1 = ConvBlock(current_channels, bottleneck_intermediate_channels)
        
        if time_embedding_dim is not None:
            self.bottleneck_time_mlp = FeatureEmbeddingMLP(time_embedding_dim, bottleneck_intermediate_channels)
        else:
            self.bottleneck_time_mlp = None
        
        if self.cond_embedding_dim is not None: 
            self.bottleneck_cond_mlp = FeatureEmbeddingMLP(self.cond_embedding_dim, bottleneck_intermediate_channels)
        else:
            self.bottleneck_cond_mlp = None
            
        self.bottleneck_conv2 = ConvBlock(bottleneck_intermediate_channels, current_channels)

        # Up blocks
        for i, mult in reversed(list(enumerate(channel_mults))):
            skip_connection_channels = base_channels * mult 
            self.ups.append(UpBlock(in_channels_up=current_channels,
                                     in_channels_skip=skip_connection_channels,
                                     out_channels=skip_connection_channels, 
                                     time_emb_dim=time_embedding_dim,
                                     cond_emb_dim=self.cond_embedding_dim)) 
            current_channels = skip_connection_channels 

        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, time_steps, conditioning_data=None): 
        t_emb = self.time_projection(time_steps) 

        c_emb = None
        if conditioning_data is not None and self.conditional_processor is not None:
            c_emb = self.conditional_processor(conditioning_data) 
        
        x = self.initial_conv(x) 

        skip_connections = []
        for down_block in self.downs:
            x, skip = down_block(x, t_emb, c_emb) 
            skip_connections.append(skip)

        x = self.bottleneck_conv1(x)
        if self.bottleneck_time_mlp is not None and t_emb is not None:
            time_info_bn = self.bottleneck_time_mlp(t_emb).unsqueeze(-1)
            x = x + time_info_bn
        if self.bottleneck_cond_mlp is not None and c_emb is not None: 
            cond_info_bn = self.bottleneck_cond_mlp(c_emb).unsqueeze(-1)
            x = x + cond_info_bn
        x = self.bottleneck_conv2(x)

        skip_connections = reversed(skip_connections) 
        for up_block, skip in zip(self.ups, skip_connections):
            x = up_block(x, skip, t_emb, c_emb) 

        return self.final_conv(x)