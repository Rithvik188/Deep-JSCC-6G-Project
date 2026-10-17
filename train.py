"""
train.py
--------
Latent-space training loop.

Inputs to each batch: pre-cached latents of shape (B, 4, 16, 16).
No VAE encode/decode during training — only ViT Transmitter + Channel + ViT Receiver.

Loss = MSE(latents_hat, latents) + 0.1 * (1 - CosineSimiliarity)

This keeps the full model quality (Rician, Rayleigh, AWGN, dynamic SNR)
while cutting epoch time from ~25 min to ~2-3 min on an RTX 3060.
"""

import torch
import torch.nn.functional as F
import random


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

        # ---- Dynamic channel conditions (same as before) ---- #
        snr_db       = random.uniform(0.0, 20.0)
        channel_type = random.choice(['awgn', 'rician', 'rayleigh'])

        # ---- Forward: Transmitter → Channel → Receiver ---- #
        # No VAE involved. latents are the direct input.
        s           = model.transmitter(latents)
        y, h        = model.channel(s, snr_db=snr_db, channel_type=channel_type)
        latents_hat = model.receiver(y, h)

        # ---- Loss ---- #
        mse_loss = F.mse_loss(latents_hat, latents)
        cos_sim  = F.cosine_similarity(
            latents_hat.flatten(1), latents.flatten(1), dim=1
        ).mean()
        loss = mse_loss + 0.1 * (1.0 - cos_sim)

        # ---- Backward ---- #
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(
                f"   -> Batch [{i+1}/{len(dataloader)}] | "
                f"Loss: {loss.item():.4f} | MSE: {mse_loss.item():.4f} | "
                f"CosSim: {cos_sim.item():.4f} | SNR: {snr_db:.1f}dB | Ch: {channel_type}",
                flush=True,
            )

        # Save checkpoint every 200 batches
        if (i + 1) % 200 == 0:
            torch.save(model.state_dict(), 'checkpoints/latest_jscc_model.pth')

    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, device='cpu', max_batches=100):
    """
    Quick latent-space evaluation: MSE between received and original latents.
    Higher cosine similarity = better channel equalization.
    """
    model.eval()
    total_mse  = 0.0
    total_cos  = 0.0
    batches    = 0

    with torch.no_grad():
        for latents in dataloader:
            if batches >= max_batches:
                break

            latents     = latents.to(device, non_blocking=True)
            s           = model.transmitter(latents)
            y, h        = model.channel(s, snr_db=10.0, channel_type='rayleigh')
            latents_hat = model.receiver(y, h)

            total_mse += F.mse_loss(latents_hat, latents).item()
            total_cos += F.cosine_similarity(
                latents_hat.flatten(1), latents.flatten(1), dim=1
            ).mean().item()
            batches += 1

    return (
        total_mse / max(batches, 1),
        total_cos / max(batches, 1),
    )
