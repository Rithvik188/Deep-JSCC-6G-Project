import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class SemanticVAE(nn.Module):
    def __init__(self, pretrained_model_name="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        # Load the pre-trained Stable Diffusion VAE
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name)
        
        # Freeze VAE weights - we only train the JSCC ViT
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # The VAE uses a scaling factor to keep latents around variance 1
        self.scaling_factor = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, x):
        """
        x: tensor of shape (B, 3, H, W) with values in [0, 1]
        Returns: latents of shape (B, 4, H/8, W/8)
        """
        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0
        
        # Get latent distribution
        posterior = self.vae.encode(x).latent_dist
        
        # We use the mode (deterministic) for JSCC encoding
        latents = posterior.mode()
        
        # Scale latents
        latents = latents * self.scaling_factor
        return latents

    def decode(self, latents):
        """
        latents: tensor of shape (B, 4, H/8, W/8)
        Returns: image of shape (B, 3, H, W) with values in [0, 1]
        """
        # Unscale latents
        latents = latents / self.scaling_factor
        
        # Decode
        image = self.vae.decode(latents).sample
        
        # Convert from [-1, 1] back to [0, 1]
        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0.0, 1.0)
        return image
