import torch
import torch.nn as nn
from models.encoder import ViTTransmitter
from models.decoder import ViTReceiver
from channel.fading import WirelessChannel
from models.vae import SemanticVAE

class DeepJSCCSystem(nn.Module):
    def __init__(self, use_vae=True):
        super().__init__()
        self.use_vae = use_vae
        if self.use_vae:
            self.vae = SemanticVAE()
            # Latents from VAE for 128x128 image are 16x16x4
            self.transmitter = ViTTransmitter(img_size=16, patch_size=2, in_channels=4)
            self.receiver = ViTReceiver(img_size=16, patch_size=2, out_channels=4, use_sigmoid=False)
        else:
            self.transmitter = ViTTransmitter()
            self.receiver = ViTReceiver()
            
        self.channel = WirelessChannel()

    def forward(self, x, snr_db=10.0, channel_type='rayleigh'):
        if self.use_vae:
            # Compress to latents
            latents = self.vae.encode(x)
            s = self.transmitter(latents)
            y, h = self.channel(s, snr_db=snr_db, channel_type=channel_type)
            latents_hat = self.receiver(y, h)
            # Reconstruct from latents
            x_hat = self.vae.decode(latents_hat)
        else:
            s = self.transmitter(x)
            y, h = self.channel(s, snr_db=snr_db, channel_type=channel_type)
            x_hat = self.receiver(y, h)
            
        return x_hat, s, y

    def forward_latent_training(self, x, snr_db=10.0, channel_type='rayleigh'):
        """Fast training pass that optimizes the latent communication directly with CSI."""
        latents = self.vae.encode(x)
        s = self.transmitter(latents)
        y, h = self.channel(s, snr_db=snr_db, channel_type=channel_type)
        latents_hat = self.receiver(y, h)
        return latents_hat, latents
