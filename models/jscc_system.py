import torch
import torch.nn as nn
from models.encoder import ViTTransmitter
from models.decoder import ViTReceiver
from channel.fading import WirelessChannel

# Bump when the transmitter/receiver change in a way that makes old checkpoints
# load without error but behave wrongly (e.g. a new receiver input definition).
# v2: MMSE-equalized receiver input + unit complex symbol power.
MODEL_VERSION = 2


class DeepJSCCSystem(nn.Module):
    """
    Semantic Deep JSCC: frozen VAE -> ViT Transmitter -> Channel -> ViT Receiver -> frozen VAE.

    use_vae=False builds only Transmitter + Channel + Receiver (used for training on
    pre-cached latents). The trainable weights and their state-dict keys are identical
    either way, so training checkpoints load straight into the full system.
    """
    def __init__(self, use_vae=True):
        super().__init__()
        self.use_vae = use_vae
        if self.use_vae:
            from models.vae import SemanticVAE  # diffusers import is slow; skip it when not needed
            self.vae = SemanticVAE()
        # VAE latents for a 128x128 image are 4x16x16
        self.transmitter = ViTTransmitter(img_size=16, patch_size=2, in_channels=4)
        self.receiver = ViTReceiver(img_size=16, patch_size=2, out_channels=4, use_sigmoid=False)
        self.channel = WirelessChannel()

    def forward_latent(self, latents, snr_db=10.0, channel_type='rayleigh'):
        """Transmitter -> Channel -> Receiver on VAE latents (B, 4, 16, 16)."""
        s = self.transmitter(latents)
        y, h = self.channel(s, snr_db=snr_db, channel_type=channel_type)
        return self.receiver(y, h, snr_db=snr_db)

    def forward(self, x, snr_db=10.0, channel_type='rayleigh'):
        """Full pipeline on images (B, 3, 128, 128) in [0, 1]. Requires use_vae=True."""
        latents = self.vae.encode(x)
        latents_hat = self.forward_latent(latents, snr_db=snr_db, channel_type=channel_type)
        return self.vae.decode(latents_hat)


def load_jscc_weights(model, checkpoint_path, device):
    """
    Loads trained transmitter/receiver weights into `model` and fails loudly on any mismatch.
    VAE weights are never in the checkpoint; they stay at the pretrained values.
    Returns the checkpoint's epoch.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not (isinstance(ckpt, dict) and 'model_state' in ckpt):
        raise RuntimeError(f"'{checkpoint_path}' is not a training checkpoint (no 'model_state').")
    version = ckpt.get('model_version', 1)
    if version != MODEL_VERSION:
        raise RuntimeError(
            f"'{checkpoint_path}' was trained with model version {version}, "
            f"but the code is version {MODEL_VERSION}. Retrain or check out the matching code."
        )
    state = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('vae.')}

    expected = {k for k in model.state_dict() if not k.startswith('vae.')}
    missing = sorted(expected - state.keys())
    unexpected = sorted(state.keys() - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' does not match the model "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys). "
            f"First missing: {missing[:3]} | first unexpected: {unexpected[:3]}"
        )
    model.load_state_dict(state, strict=False)
    return ckpt.get('epoch')
