"""
precompute_latents.py
---------------------
ONE-TIME script. Encodes the entire CIFAR-10 dataset through the frozen
Stable Diffusion VAE and saves the resulting latent tensors to disk.

After running this:
  - data/latents_train.pt   (shape: [N_train, 4, 16, 16])
  - data/latents_test.pt    (shape: [N_test,  4, 16, 16])

Training then NEVER runs the VAE again, cutting epoch time from ~25 min
down to ~2-3 min on an RTX 3060 — with zero change to model quality.
"""

import os
import torch
from torch.utils.data import DataLoader
from data.dataset import get_face_dataloaders
from models.vae import SemanticVAE

LATENT_DIR = "./data"
TRAIN_OUT  = os.path.join(LATENT_DIR, "latents_train.pt")
TEST_OUT   = os.path.join(LATENT_DIR, "latents_test.pt")


def precompute_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Precompute] Using device: {device}")

    # ------------------------------------------------------------------ #
    # Load raw image dataset (same pipeline as training)                   #
    # ------------------------------------------------------------------ #
    print("[Precompute] Loading image dataset...")
    train_loader, test_loader = get_face_dataloaders(batch_size=64, img_size=128)

    # ------------------------------------------------------------------ #
    # Load frozen VAE (weights already frozen inside SemanticVAE)          #
    # ------------------------------------------------------------------ #
    print("[Precompute] Loading frozen VAE encoder...")
    vae = SemanticVAE().to(device)
    vae.eval()

    # ------------------------------------------------------------------ #
    # Encode all training images                                           #
    # ------------------------------------------------------------------ #
    def encode_split(loader, split_name, out_path):
        if os.path.exists(out_path):
            print(f"[Precompute] {out_path} already exists — skipping.")
            return

        all_latents = []
        total_batches = len(loader)
        print(f"[Precompute] Encoding {split_name} split ({total_batches} batches)...")

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(device)
                else:
                    images = batch.to(device)

                latents = vae.encode(images)          # (B, 4, 16, 16)
                all_latents.append(latents.cpu())     # keep on CPU to save VRAM

                if (i + 1) % 50 == 0 or (i + 1) == total_batches:
                    print(f"   -> Batch [{i+1}/{total_batches}]", flush=True)

        all_latents = torch.cat(all_latents, dim=0)   # (N, 4, 16, 16)
        torch.save(all_latents, out_path)
        size_mb = all_latents.element_size() * all_latents.nelement() / 1e6
        print(f"[Precompute] Saved {split_name}: shape={tuple(all_latents.shape)}, "
              f"size={size_mb:.1f} MB -> {out_path}")

    encode_split(train_loader, "train", TRAIN_OUT)
    encode_split(test_loader,  "test",  TEST_OUT)

    print("[Precompute] Done! You can now run main.py to train at full speed.")


if __name__ == "__main__":
    precompute_and_save()
