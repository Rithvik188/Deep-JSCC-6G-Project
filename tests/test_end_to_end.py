import torch
import torch.nn.functional as F
import pytest
from models.jscc_system import DeepJSCCSystem

def test_end_to_end_pipeline():
    # Initialize system
    system = DeepJSCCSystem()
    
    # Mock image batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    
    # Forward pass
    x_hat, s, y = system(x, snr_db=10.0, channel_type='rayleigh')
    
    # Verify output shape
    expected_shape = (batch_size, 3, 32, 32)
    assert x_hat.shape == expected_shape, f"Expected shape {expected_shape}, got {x_hat.shape}"
    
    # Verify bounded [0, 1]
    assert torch.all(x_hat >= 0.0) and torch.all(x_hat <= 1.0), "Output values are not in [0, 1] range"
    
    # Compute dummy loss
    loss = F.mse_loss(x_hat, x)
    
    # Backward pass
    loss.backward()
    
    # Check encoder gradients
    encoder_grad = False
    for param in system.transmitter.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            encoder_grad = True
            break
            
    # Check decoder gradients
    decoder_grad = False
    for param in system.receiver.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            decoder_grad = True
            break
            
    assert encoder_grad, "Encoder parameters did not receive gradients"
    assert decoder_grad, "Decoder parameters did not receive gradients"
