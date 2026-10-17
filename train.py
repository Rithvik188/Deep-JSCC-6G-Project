"""
train.py
--------
Latent-space training loop.

Inputs to each batch: pre-cached latents of shape (B, 4, 16, 16).
No VAE encode/decode during training — only ViT Transmitter + Channel + ViT Receiver.

Loss = MSE(latents_hat, latents) + 0.1 * (1 - CosineSimilarity)
"""

import torch
import torch.nn.functional as F
import random

CHANNELS = ['awgn', 'rician', 'rayleigh']
SNR_RANGE = (0.0, 20.0)


def jscc_loss(latents_hat, latents):
    mse_loss = F.mse_loss(latents_hat, latents)
    cos_sim = F.cosine_similarity(latents_hat.flatten(1), latents.flatten(1), dim=1).mean()
    return mse_loss + 0.1 * (1.0 - cos_sim), mse_loss, cos_sim


def train_epoch(model, dataloader, optimizer, device='cpu'):
    """
    Runs one full epoch of latent-space JSCC training.
    `dataloader` must yield raw latent tensors (B, 4, 16, 16).
    """
    model.train()
    total_loss = 0.0

    for i, latents in enumerate(dataloader):
        latents = latents.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ---- Dynamic channel conditions: random channel per batch, random SNR per sample ---- #
        snr_db = torch.empty(latents.size(0), device=device).uniform_(*SNR_RANGE)
        channel_type = random.choice(CHANNELS)

        latents_hat = model.forward_latent(latents, snr_db=snr_db, channel_type=channel_type)
        loss, mse_loss, cos_sim = jscc_loss(latents_hat, latents)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(
                f"   -> Batch [{i+1}/{len(dataloader)}] | "
                f"Loss: {loss.item():.4f} | MSE: {mse_loss.item():.4f} | "
                f"CosSim: {cos_sim.item():.4f} | Ch: {channel_type}",
                flush=True,
            )

    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, device='cpu', snr_db=10.0, max_batches=100):
    """
    Latent-space evaluation at a fixed SNR for every channel type.
    Returns {channel: (mse, cos_sim)}. Uses a fixed seed so epochs are comparable.
    """
    model.eval()
    results = {}
    gen_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    with torch.no_grad():
        for channel_type in CHANNELS:
            torch.manual_seed(1234)
            total_mse, total_cos, batches = 0.0, 0.0, 0
            for latents in dataloader:
                if batches >= max_batches:
                    break
                latents = latents.to(device, non_blocking=True)
                latents_hat = model.forward_latent(latents, snr_db=snr_db, channel_type=channel_type)
                _, mse_loss, cos_sim = jscc_loss(latents_hat, latents)
                total_mse += mse_loss.item()
                total_cos += cos_sim.item()
                batches += 1
            results[channel_type] = (total_mse / max(batches, 1), total_cos / max(batches, 1))

    torch.random.set_rng_state(gen_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return results
