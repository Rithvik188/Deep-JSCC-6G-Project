import math
import torch
from baselines.digital import capacity_bits_per_symbol, bit_budget, digital_transmit, FAILED_FRAME_VALUE


def test_awgn_capacity_matches_shannon():
    assert math.isclose(capacity_bits_per_symbol(10.0, 'awgn'), math.log2(11.0))
    assert math.isclose(bit_budget(256, 0.0, 'awgn'), 256.0)


def test_fading_capacity_below_awgn():
    # Jensen: ergodic fading capacity with E|h|^2 = 1 is below the AWGN capacity
    for ch in ['rician', 'rayleigh']:
        assert capacity_bits_per_symbol(10.0, ch) < capacity_bits_per_symbol(10.0, 'awgn')
    assert capacity_bits_per_symbol(10.0, 'rayleigh') < capacity_bits_per_symbol(10.0, 'rician')


def test_designed_digital_has_cliff():
    torch.manual_seed(0)
    img = torch.rand(3, 128, 128) * 0.2 + 0.4
    below, nb_below = digital_transmit(img, 1024, 5.0, 'awgn', design_snr_db=10.0)
    above, nb_above = digital_transmit(img, 1024, 20.0, 'awgn', design_snr_db=10.0)
    at, nb_at = digital_transmit(img, 1024, 10.0, 'awgn', design_snr_db=10.0)
    assert nb_below == 0 and torch.all(below == FAILED_FRAME_VALUE)
    assert nb_at > 0 and nb_at <= bit_budget(1024, 10.0, 'awgn') / 8
    assert torch.equal(above, at)  # no gain above the design SNR
