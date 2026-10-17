import torch
import torch.nn as nn
import math


def snr_to_noise_var(snr_db, x):
    """
    snr_db: float or tensor of shape (B,). Returns noise variance per complex symbol,
    shaped (B, 1) so it broadcasts over the symbol dimension of x (B, K).
    """
    snr_db = torch.as_tensor(snr_db, dtype=x.dtype, device=x.device)
    if snr_db.dim() == 0:
        snr_db = snr_db.expand(x.size(0))
    return (10.0 ** (-snr_db / 10.0)).unsqueeze(1)


class WirelessChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, snr_db, channel_type='awgn'):
        """
        x: shape (B, 2, K) where index 0 is Real, 1 is Imaginary.
           Assumed to have average complex symbol power E|x|^2 = 1.
        snr_db: Signal-to-Noise Ratio in dB, float or per-sample tensor of shape (B,)
        channel_type: 'awgn', 'rayleigh', or 'rician'
        """
        x_real = x[:, 0, :]
        x_imag = x[:, 1, :]

        # Complex noise with total variance noise_var per symbol (noise_var/2 per component)
        noise_var = snr_to_noise_var(snr_db, x_real)
        noise_std = torch.sqrt(noise_var / 2.0)

        n_real = torch.randn_like(x_real) * noise_std
        n_imag = torch.randn_like(x_imag) * noise_std

        # Fading coefficient generation (i.i.d. per symbol, E|h|^2 = 1)
        if channel_type == 'awgn':
            h_real = torch.ones_like(x_real)
            h_imag = torch.zeros_like(x_imag)
        elif channel_type == 'rayleigh':
            std_rayleigh = math.sqrt(0.5)
            h_real = torch.randn_like(x_real) * std_rayleigh
            h_imag = torch.randn_like(x_imag) * std_rayleigh
        elif channel_type == 'rician':
            k_factor = 3.0
            mean_real = math.sqrt(k_factor / (k_factor + 1.0))
            mean_imag = 0.0
            std_scatter = math.sqrt(1.0 / (2.0 * (k_factor + 1.0)))

            h_real = mean_real + torch.randn_like(x_real) * std_scatter
            h_imag = mean_imag + torch.randn_like(x_imag) * std_scatter
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")

        # Manual complex multiplication and noise addition
        y_real = h_real * x_real - h_imag * x_imag + n_real
        y_imag = h_real * x_imag + h_imag * x_real + n_imag

        # Stack back to (B, 2, K)
        y = torch.stack([y_real, y_imag], dim=1)
        h = torch.stack([h_real, h_imag], dim=1)

        return y, h
