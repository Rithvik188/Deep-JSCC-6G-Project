"""
main.py
-------
Entry point for Deep JSCC 6G training.

Step 1 (auto): If cached latents don't exist, runs precompute_latents.py first.
Step 2:        Trains only ViT Transmitter + Channel + ViT Receiver on cached latents.

Training is resumable: a full checkpoint (model, optimizer, scheduler, epoch) is written to
checkpoints/latest_jscc_model.pth after every epoch, and `python main.py` continues from it.
The best epoch (lowest validation MSE averaged over AWGN/Rician/Rayleigh at 10 dB) is also
copied to checkpoints/best_jscc_model.pth, which is what server.py and evaluate_visuals.py load.
"""

import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.latent_dataset import get_latent_dataloaders
from models.jscc_system import DeepJSCCSystem, MODEL_VERSION
from train import train_epoch, evaluate_epoch

LATENT_TRAIN = "./data/latents_train.pt"
LATENT_TEST  = "./data/latents_test.pt"
LATEST_CKPT  = "checkpoints/latest_jscc_model.pth"
BEST_CKPT    = "checkpoints/best_jscc_model.pth"
NUM_EPOCHS   = 30
BATCH_SIZE   = 64


def maybe_precompute():
    """Auto-run precompute if cached latents are missing."""
    if not os.path.exists(LATENT_TRAIN) or not os.path.exists(LATENT_TEST):
        print("=" * 60)
        print("Cached latents not found. Running one-time precomputation.")
        print("=" * 60)
        from precompute_latents import precompute_and_save
        precompute_and_save()
        print("=" * 60)
        print("Precomputation complete. Starting training...")
        print("=" * 60)


def try_resume(model, optimizer, scheduler, device):
    """Resumes from the latest checkpoint. Returns (start_epoch, best_mse)."""
    if not os.path.exists(LATEST_CKPT):
        return 0, float("inf")

    ckpt = torch.load(LATEST_CKPT, map_location=device, weights_only=True)
    version = ckpt.get("model_version", 1) if isinstance(ckpt, dict) else 1
    if version != MODEL_VERSION or "scheduler_state" not in ckpt:
        raise RuntimeError(
            f"'{LATEST_CKPT}' is from an older model version ({version}) and cannot be resumed. "
            f"Move it out of checkpoints/ to start a fresh run."
        )

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from '{LATEST_CKPT}' -> continuing from Epoch {start_epoch + 1}")
    return start_epoch, ckpt["best_mse"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    maybe_precompute()

    print("Loading cached latent dataset...")
    train_loader, test_loader = get_latent_dataloaders(
        train_path=LATENT_TRAIN,
        test_path=LATENT_TEST,
        batch_size=BATCH_SIZE,
    )

    # Transmitter + Channel + Receiver only; the VAE is used by server.py at inference
    print("Initializing JSCC ViT model (Transmitter + Channel + Receiver)...")
    model = DeepJSCCSystem(use_vae=False).to(device)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    start_epoch, best_mse = try_resume(model, optimizer, scheduler, device)

    print(f"\nStarting Training: Epoch {start_epoch + 1} -> {NUM_EPOCHS}")
    print(f"Batches per epoch: {len(train_loader)} | Batch size: {BATCH_SIZE}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH [{epoch+1}/{NUM_EPOCHS}]  (lr={scheduler.get_last_lr()[0]:.2e})")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val = evaluate_epoch(model, test_loader, device)
        val_mse = sum(mse for mse, _ in val.values()) / len(val)

        scheduler.step()

        per_channel = " | ".join(f"{ch} {mse:.4f}" for ch, (mse, _) in val.items())
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary: Train Loss={train_loss:.4f} | "
              f"Val MSE@10dB: {per_channel} | mean {val_mse:.4f}")

        is_best = val_mse < best_mse
        if is_best:
            best_mse = val_mse

        save_state = {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch":           epoch,
            "best_mse":        best_mse,
            "val":             val,
            "model_version":   MODEL_VERSION,
        }
        torch.save(save_state, LATEST_CKPT)
        if is_best:
            torch.save(save_state, BEST_CKPT)
            print(f" -> Best checkpoint saved! (mean Val MSE={best_mse:.4f})")


if __name__ == "__main__":
    main()
