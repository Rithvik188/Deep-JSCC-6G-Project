"""
data/latent_dataset.py
----------------------
Serves pre-cached VAE latents directly from disk.
No image loading, no VAE encode — pure tensor serving.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class LatentDataset(Dataset):
    """
    Wraps a pre-computed latent tensor file.
    Each item is a single latent of shape (4, 16, 16).
    """
    def __init__(self, latent_path: str):
        self.latents = torch.load(latent_path, weights_only=True)  # (N, 4, 16, 16)
        print(f"[LatentDataset] Loaded {len(self.latents)} latents from {latent_path} "
              f"| shape per sample: {tuple(self.latents.shape[1:])}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]   # (4, 16, 16) — no label needed


def get_latent_dataloaders(train_path: str, test_path: str, batch_size: int = 64):
    """
    Returns (train_loader, test_loader) that yield latent tensors directly.
    Drop-in replacement for get_face_dataloaders during training.
    """
    train_dataset = LatentDataset(train_path)
    test_dataset  = LatentDataset(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,   # faster GPU transfer
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, test_loader
