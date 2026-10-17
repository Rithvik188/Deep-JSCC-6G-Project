import torch
import torch.nn as nn

class ChannelPowerConstraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Average power of 1.0. x is (Batch, 2, num_symbols)
        # Normalize per item in the batch.
        rms = torch.sqrt(torch.mean(x**2, dim=(1, 2), keepdim=True))
        return x / (rms + 1e-8)

class ViTTransmitter(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128, depth=4, num_heads=4, num_symbols=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Linear projection head
        self.head = nn.Linear(self.num_patches * embed_dim, 2 * num_symbols)
        
        # Power Constraint
        self.power_constraint = ChannelPowerConstraint()

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (Batch, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (Batch, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)  # (Batch, num_patches, embed_dim)
        
        # Flatten and Project
        batch_size = x.size(0)
        x = x.flatten(1)  # (Batch, num_patches * embed_dim)
        x = self.head(x)  # (Batch, 2 * num_symbols)
        x = x.view(batch_size, 2, -1)  # (Batch, 2, num_symbols)
        
        # Power Constraint
        x = self.power_constraint(x)
        return x
