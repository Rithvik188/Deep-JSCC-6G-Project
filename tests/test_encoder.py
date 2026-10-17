import torch
import pytest
from models.encoder import ViTTransmitter

def test_vit_transmitter():
    batch_size = 8
    img_size = 32
    in_channels = 3
    num_symbols = 256
    
    # Mock image batch
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    model = ViTTransmitter(
        img_size=img_size, 
        patch_size=4, 
        in_channels=in_channels, 
        embed_dim=128, 
        depth=4, 
        num_heads=4, 
        num_symbols=num_symbols
    )
    
    out = model(x)
    
    # Assert output shape
    assert out.shape == (batch_size, 2, num_symbols), f"Expected shape {(batch_size, 2, num_symbols)}, got {out.shape}"
    
    # Assert average power is 1.0 +/- 1e-4
    power = torch.mean(out**2).item()
    assert abs(power - 1.0) <= 1e-4, f"Average power is {power}, expected 1.0"
