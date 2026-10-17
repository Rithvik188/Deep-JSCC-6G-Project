import torch
import torch.nn as nn

class ViTReceiver(nn.Module):
    def __init__(self, img_size=32, patch_size=4, out_channels=3, embed_dim=128, depth=4, num_heads=4, num_symbols=256, use_sigmoid=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Project received symbols (B, 2, num_symbols) and CSI (B, 2, num_symbols) -> total 4 * num_symbols
        # If h is provided, input is (B, 4, num_symbols); otherwise fallback to zero CSI
        self.input_proj = nn.Linear(4 * num_symbols, self.num_patches * embed_dim)
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer Decoder/Encoder Layers
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
        self.head = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, y, h=None):
        # y: (Batch, 2, num_symbols), h: (Batch, 2, num_symbols)
        batch_size = y.size(0)
        
        if h is None:
            # Fallback for AWGN without explicit h: assume h = 1 + 0j
            h_real = torch.ones_like(y[:, 0:1, :])
            h_imag = torch.zeros_like(y[:, 1:2, :])
            h = torch.cat([h_real, h_imag], dim=1)
            
        # Concatenate received symbols y and CSI h -> (Batch, 4, num_symbols)
        yh = torch.cat([y, h], dim=1)
        yh = yh.view(batch_size, -1)  # (Batch, 4 * num_symbols)
        
        x = self.input_proj(yh)  # (Batch, num_patches * embed_dim)
        x = x.view(batch_size, self.num_patches, self.embed_dim)  # (Batch, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)  # (Batch, num_patches, embed_dim)
        
        # Project back to patch pixels
        x = self.head(x)  # (Batch, num_patches, patch_size * patch_size * out_channels)
        
        # Reshape and rearrange to full image
        h_patches = self.img_size // self.patch_size
        w_patches = self.img_size // self.patch_size
        x = x.view(batch_size, h_patches, w_patches, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, self.out_channels, self.img_size, self.img_size)
        
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x
