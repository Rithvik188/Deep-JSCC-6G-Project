import math
import torch
import pytest
from channel.fading import WirelessChannel


@pytest.mark.parametrize("channel_type", ["awgn", "rician", "rayleigh"])
def test_wireless_channel_shapes(channel_type):
    x = torch.randn(16, 2, 256)
    y, h = WirelessChannel()(x, snr_db=10, channel_type=channel_type)
    assert y.shape == (16, 2, 256)
    assert h.shape == (16, 2, 256)


@pytest.mark.parametrize("channel_type", ["rician", "rayleigh"])
def test_fading_has_unit_average_gain(channel_type):
    x = torch.zeros(64, 2, 4096)
    _, h = WirelessChannel()(x, snr_db=100, channel_type=channel_type)
    gain = (h ** 2).sum(dim=1).mean().item()
    assert abs(gain - 1.0) < 0.02


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_noise_power_matches_snr(snr_db):
    # Zero input on AWGN: the output is pure noise with power 10^(-SNR/10) per complex symbol
    x = torch.zeros(64, 2, 4096)
    y, _ = WirelessChannel()(x, snr_db=snr_db, channel_type="awgn")
    noise_power = (y ** 2).sum(dim=1).mean().item()
    expected = 10 ** (-snr_db / 10)
    assert math.isclose(noise_power, expected, rel_tol=0.02)


def test_per_sample_snr():
    x = torch.zeros(2, 2, 8192)
    y, _ = WirelessChannel()(x, snr_db=torch.tensor([0.0, 20.0]), channel_type="awgn")
    p = (y ** 2).sum(dim=1).mean(dim=1)
    assert math.isclose(p[0].item(), 1.0, rel_tol=0.05)
    assert math.isclose(p[1].item(), 0.01, rel_tol=0.05)
