import torch
import pytest
from models.encoder import ViTTransmitter


def test_vit_transmitter():
    batch_size = 8
    num_symbols = 256

    # Mock VAE latent batch (B, 4, 16, 16)
    x = torch.randn(batch_size, 4, 16, 16)

    model = ViTTransmitter(
        img_size=16,
        patch_size=2,
        in_channels=4,
        embed_dim=128,
        depth=4,
        num_heads=4,
        num_symbols=num_symbols
    )

    out = model(x)

    # Assert output shape
    assert out.shape == (batch_size, 2, num_symbols), f"Expected shape {(batch_size, 2, num_symbols)}, got {out.shape}"

    # Assert average complex symbol power E|s|^2 is 1.0 for every item
    power = (out ** 2).sum(dim=1).mean(dim=1)
    assert torch.allclose(power, torch.ones(batch_size), atol=1e-4), f"Per-item complex power {power}, expected 1.0"
