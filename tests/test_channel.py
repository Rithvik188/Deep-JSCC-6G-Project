import torch
import pytest
from channel.fading import WirelessChannel

def test_wireless_channel():
    batch_size = 16
    channels = 2
    num_symbols = 256
    
    # Mock transmitted tensor
    x = torch.randn(batch_size, channels, num_symbols)
    
    # Initialize channel
    channel = WirelessChannel()
    
    # Pass through rayleigh channel
    y = channel(x, snr_db=10, channel_type='rayleigh')
    
    # Assert output shape remains (16, 2, 256)
    expected_shape = (batch_size, channels, num_symbols)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"
