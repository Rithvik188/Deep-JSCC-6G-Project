import torch
import torch.nn as nn
from channel.fading import snr_to_noise_var

class ViTReceiver(nn.Module):
    def __init__(self, img_size=32, patch_size=4, out_channels=3, embed_dim=128, depth=4, num_heads=4, num_symbols=256, use_sigmoid=True, receiver_mode='mmse'):
        super().__init__()
        if receiver_mode not in ('mmse', 'raw'):
            raise ValueError(f"Unknown receiver_mode: {receiver_mode}")
        self.receiver_mode = receiver_mode
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Input features per symbol (4 * num_symbols total):
        #   'mmse': MMSE-equalized symbol (real, imag), channel gain |h|^2, and SNR.
        #   'raw':  received y (real, imag) and CSI h (real, imag), concatenated.
        # 'mmse' is the default: undoing fading needs a per-symbol complex division,
        # which a dense layer over raw [y, h] cannot represent. 'raw' exists only
        # for the ablation that demonstrates this.
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

    def forward(self, y, h=None, snr_db=10.0):
        # y: (Batch, 2, num_symbols), h: (Batch, 2, num_symbols)
        # snr_db: float or per-sample tensor (Batch,), known at the receiver like the CSI
        batch_size = y.size(0)
        y_real, y_imag = y[:, 0, :], y[:, 1, :]

        if h is None:
            # Fallback for AWGN without explicit h: assume h = 1 + 0j
            h_real = torch.ones_like(y_real)
            h_imag = torch.zeros_like(y_imag)
        else:
            h_real, h_imag = h[:, 0, :], h[:, 1, :]

        if self.receiver_mode == 'raw':
            feats = torch.stack([y_real, y_imag, h_real, h_imag], dim=1)
            return self._decode(feats.view(batch_size, -1))

        # Per-symbol MMSE equalization: x_eq = y * conj(h) / (|h|^2 + noise_var)
        noise_var = snr_to_noise_var(snr_db, y_real)        # (Batch, 1)
        gain = h_real ** 2 + h_imag ** 2                     # |h|^2
        denom = gain + noise_var
        x_eq_real = (y_real * h_real + y_imag * h_imag) / denom
        x_eq_imag = (y_imag * h_real - y_real * h_imag) / denom
        snr_feat = (-10.0 * torch.log10(noise_var) / 20.0).expand_as(gain)  # SNR scaled to ~[0, 1]

        feats = torch.stack([x_eq_real, x_eq_imag, gain, snr_feat], dim=1)  # (Batch, 4, num_symbols)
        return self._decode(feats.view(batch_size, -1))  # (Batch, 4 * num_symbols)

    def _decode(self, feats):
        batch_size = feats.size(0)
        x = self.input_proj(feats)  # (Batch, num_patches * embed_dim)
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
