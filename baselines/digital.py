"""
baselines/digital.py
--------------------
Separate source-channel coding baseline ("digital"): compress the frame with WebP and send
the bits over an IDEAL capacity-achieving channel code using the same number of complex
channel symbols K as the Deep JSCC model.

This is an optimistic upper bound for any practical digital scheme (real LDPC/polar codes
need a margin above capacity and finite block lengths):
  - Bit budget per frame = K * C(SNR) bits.
    AWGN:   C = log2(1 + SNR)
    Fading: ergodic capacity C = E[log2(1 + |h|^2 SNR)] with perfect receiver CSI,
            using the same fading distributions as channel/fading.py.
  - The encoder tries several resolutions and WebP qualities and keeps the best-PSNR version
    that fits the budget (sending a small sharp image and upscaling beats over-compressing
    the full-size one at low budgets).
  - "Designed" mode reproduces the cliff effect: the rate is chosen for a design SNR; if the
    actual SNR is lower the code cannot be decoded and nothing usable arrives, if it is higher
    the quality does not improve.

WebP is used instead of JPEG because JPEG's ~600-byte header alone exceeds the budget at
small K.
"""

import io
import math
import functools
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from channel.fading import WirelessChannel

RESOLUTIONS = (128, 96, 64, 48, 32)
FAILED_FRAME_VALUE = 0.5  # what the receiver shows when decoding fails (flat gray)

# "Practical" digital: a discrete set of modulation-and-coding rates (bits per complex symbol,
# e.g. QPSK 1/8 ... 64-QAM 5/6), each usable only when the SNR is GAP_DB above the Shannon
# limit for that rate. This approximates real LDPC/polar codes at moderate block lengths.
MCS_RATES = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
GAP_DB = 2.0


@functools.lru_cache(maxsize=None)
def capacity_bits_per_symbol(snr_db, channel_type, num_samples=400_000):
    """Shannon capacity (bits per complex symbol) at snr_db for the given channel type."""
    snr = 10.0 ** (snr_db / 10.0)
    if channel_type == 'awgn':
        return math.log2(1.0 + snr)
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    zeros = torch.zeros(1, 2, num_samples)
    _, h = WirelessChannel()(zeros, snr_db=snr_db, channel_type=channel_type)
    torch.random.set_rng_state(gen_state)
    gain = (h ** 2).sum(dim=1)
    return torch.log2(1.0 + gain * snr).mean().item()


def rate_bits_per_symbol(snr_db, channel_type, mode='ideal'):
    """Bits per complex symbol the digital link can carry: capacity ('ideal') or the best MCS
    rate that fits under capacity with a GAP_DB margin ('practical')."""
    if mode == 'ideal':
        return capacity_bits_per_symbol(float(snr_db), channel_type)
    if mode == 'practical':
        usable = capacity_bits_per_symbol(float(snr_db) - GAP_DB, channel_type)
        return max((r for r in MCS_RATES if r <= usable), default=0.0)
    raise ValueError(f"Unknown mode: {mode}")


def bit_budget(num_symbols, snr_db, channel_type, mode='ideal'):
    return num_symbols * rate_bits_per_symbol(snr_db, channel_type, mode)


def _webp_bytes(pil_img, quality):
    buf = io.BytesIO()
    pil_img.save(buf, format='WEBP', quality=quality, method=6)
    return buf.getvalue()


def _best_fit_at_resolution(pil_img, max_bytes):
    """Highest WebP quality whose encoding fits max_bytes, or None. Binary search over quality."""
    lo, hi, best = 0, 100, None
    while lo <= hi:
        q = (lo + hi) // 2
        data = _webp_bytes(pil_img, q)
        if len(data) <= max_bytes:
            best, lo = data, q + 1
        else:
            hi = q - 1
    return best


def digital_reconstruct(img, budget_bits):
    """
    img: tensor (3, H, W) in [0, 1]. Returns (reconstruction tensor (3, H, W), bytes_used).
    If nothing fits the budget, returns a flat gray frame and 0 bytes.
    """
    _, height, width = img.shape
    max_bytes = int(budget_bits // 8)
    pil = TF.to_pil_image(img.clamp(0, 1).cpu())
    best, best_mse, best_bytes = None, float('inf'), 0

    for res in RESOLUTIONS:
        small = pil if res == width else pil.resize((res, res), Image.BICUBIC)
        data = _best_fit_at_resolution(small, max_bytes)
        if data is None:
            continue
        decoded = TF.to_tensor(Image.open(io.BytesIO(data)).convert('RGB')).unsqueeze(0)
        if res != width:
            decoded = F.interpolate(decoded, size=(height, width), mode='bicubic', align_corners=False)
        decoded = decoded.squeeze(0).clamp(0, 1)
        mse = F.mse_loss(decoded, img.cpu()).item()
        if mse < best_mse:
            best, best_mse, best_bytes = decoded, mse, len(data)

    if best is None:
        return torch.full_like(img.cpu(), FAILED_FRAME_VALUE), 0
    return best, best_bytes


def digital_transmit(img, num_symbols, snr_db, channel_type, design_snr_db=None, mode='ideal'):
    """
    Digital baseline for one frame.
      design_snr_db=None: adaptive digital (rate re-chosen for the actual SNR).
      design_snr_db=d:    rate chosen for SNR d; fails (gray frame) when the actual SNR cannot
                          support that rate, no gain above it (cliff effect).
      mode: 'ideal' (Shannon capacity) or 'practical' (discrete MCS with a GAP_DB margin).
    Returns (reconstruction (3, H, W), bytes_used).
    """
    if design_snr_db is None:
        return digital_reconstruct(img, bit_budget(num_symbols, snr_db, channel_type, mode))
    design_rate = rate_bits_per_symbol(design_snr_db, channel_type, mode)
    actual_capacity = capacity_bits_per_symbol(float(snr_db) - (GAP_DB if mode == 'practical' else 0.0), channel_type)
    if actual_capacity < design_rate:
        return torch.full_like(img.cpu(), FAILED_FRAME_VALUE), 0
    return digital_reconstruct(img, num_symbols * design_rate)
