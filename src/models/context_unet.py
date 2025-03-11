import torch
import torch.nn as nn
from embeddings.transformer import TEConditionalEmbedding

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels
        

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)
        
        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x

    
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [ResidualConvBlock(in_channels, out_channels), ResidualConvBlock(out_channels, out_channels), nn.MaxPool2d(2)]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=64, cond_model = "mlp", latent_dim=256):  # cfeat - context features
        super(ContextUnet, self).__init__()

        self.channels = n_cfeat
        self.context_size = None
        
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height
        self.cond_model = cond_model
        self.latent_dim = latent_dim

        self.init_conv = ResidualConvBlock(in_channels, n_feat)

        # Only two down-sampling layers
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 4 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((self.h//8)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 4 * n_feat)
        self.timeembed2 = EmbedFC(1, 2 * n_feat)
        
        assert self.cond_model in {"mlp", "te", "stft"}, "Chosen conditioning model was not valid, the options are mlp, te and stft"

        
        self.contextembed1 = EmbedFC(n_cfeat, 4 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 2 * n_feat)
        
        if self.cond_model == "te":
            self.pre_embed = TEConditionalEmbedding(features = n_cfeat)

        # Only two up-sampling layers
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, self.h//8, self.h//8),
            nn.GroupNorm(8, 4 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(8 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(3 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
        
        self.te_fc = nn.Linear(self.latent_dim ,self.n_cfeat)

    def forward(self, x, t, c=None):
        #print(f"input shape: {x.shape}")
        x = self.init_conv(x)
        #print(f"init conv shape: {x.shape}")
        down1 = self.down1(x)
        #print(f"down1 shape: {down1.shape}")
        down2 = self.down2(down1)
        #print(f"down2 shape: {down2.shape}")
        hiddenvec = self.to_vec(down2)
        #print(f"hiddenvec shape: {hiddenvec.shape}")
        
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x.device)

        if c is not None and self.cond_model == "te":
            c = c.reshape(x.shape[0], self.n_cfeat, 1)
            c = self.pre_embed(c).squeeze()
            c = self.te_fc(c)
        
        # Embeddings
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 4, 1, 1)
        #print(f"cemb1 shape: {cemb1.shape}")
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 4, 1, 1)
        #print(f"temb1 shape: {temb1.shape}")
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 2, 1, 1)
        #print(f"cemb2 shape: {cemb2.shape}")
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)
        #print(f"temb2 shape: {temb2.shape}")

        # Upsampling
        up1 = self.up0(hiddenvec)
        #print(f"up1 shape: {up1.shape}")
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        #print(f"up2 shape: {up2.shape}")
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        #print(f"up3 shape: {up3.shape}")
        out = self.out(torch.cat((up3, x), 1))

        return out