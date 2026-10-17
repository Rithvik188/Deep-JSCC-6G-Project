"""
main.py
-------
Entry point for Deep JSCC 6G training.

Step 1 (auto): If cached latents don't exist, runs precompute_latents.py first.
Step 2:        Trains only ViT Transmitter + Channel + ViT Receiver on cached latents.
               Epoch time: ~2-3 minutes on RTX 3060 (vs ~25 min before).
"""

import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.latent_dataset import get_latent_dataloaders
from models.jscc_system import DeepJSCCSystem
from train import train_epoch, evaluate_epoch

LATENT_TRAIN = "./data/latents_train.pt"
LATENT_TEST  = "./data/latents_test.pt"
NUM_EPOCHS   = 30   # ~2-3 min/epoch = ~60-90 min total; adjust as needed


def maybe_precompute():
    """Auto-run precompute if cached latents are missing."""
    if not os.path.exists(LATENT_TRAIN) or not os.path.exists(LATENT_TEST):
        print("=" * 60)
        print("Cached latents not found. Running one-time precomputation.")
        print("This takes ~5 minutes and only runs ONCE ever.")
        print("=" * 60)
        from precompute_latents import precompute_and_save
        precompute_and_save()
        print("=" * 60)
        print("Precomputation complete. Starting training...")
        print("=" * 60)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: Ensure cached latents exist                                  #
    # ------------------------------------------------------------------ #
    maybe_precompute()

    # ------------------------------------------------------------------ #
    # Step 2: Load cached latent dataloaders (no images, no VAE)           #
    # ------------------------------------------------------------------ #
    print("Loading cached latent dataset...")
    train_loader, test_loader = get_latent_dataloaders(
        train_path=LATENT_TRAIN,
        test_path=LATENT_TEST,
        batch_size=64,   # larger batch is fine now (no VAE VRAM overhead)
    )

    # ------------------------------------------------------------------ #
    # Step 3: Build model — only ViT Transmitter + Channel + ViT Receiver  #
    # VAE is NOT loaded here; it's only used by server.py at inference      #
    # ------------------------------------------------------------------ #
    print("Initializing JSCC ViT model (Transmitter + Channel + Receiver)...")
    model = DeepJSCCSystem(use_vae=False)  # No VAE during training
    # Override: manually build with latent-sized ViTs
    from models.encoder import ViTTransmitter
    from models.decoder import ViTReceiver
    from channel.fading import WirelessChannel
    import torch.nn as nn

    class LatentJSCC(nn.Module):
        """Lean training model: ViTTransmitter -> Channel -> ViTReceiver."""
        def __init__(self):
            super().__init__()
            self.transmitter = ViTTransmitter(img_size=16, patch_size=2, in_channels=4)
            self.receiver    = ViTReceiver(img_size=16, patch_size=2, out_channels=4, use_sigmoid=False)
            self.channel     = WirelessChannel()

    model = LatentJSCC().to(device)

    # ------------------------------------------------------------------ #
    # Step 4: Optimizer + Scheduler                                        #
    # ------------------------------------------------------------------ #
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    # ------------------------------------------------------------------ #
    # Step 5: Resume from checkpoint if available                          #
    # ------------------------------------------------------------------ #
    start_epoch = 0
    best_mse    = float("inf")

    for ckpt_path in ["checkpoints/best_jscc_model.pth", "checkpoints/latest_jscc_model.pth"]:
        if os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    model.load_state_dict(checkpoint["model_state"])
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    start_epoch = checkpoint.get("epoch", 0) + 1
                    best_mse    = checkpoint.get("best_mse", float("inf"))
                    print(f"Resumed from checkpoint '{ckpt_path}' — continuing from Epoch {start_epoch + 1}")
                else:
                    # Try loading as plain state dict (older checkpoints may be full system)
                    # Extract only matching keys
                    full_state   = checkpoint if not isinstance(checkpoint, dict) else checkpoint
                    model_keys   = set(model.state_dict().keys())
                    filtered     = {k: v for k, v in full_state.items() if k in model_keys}
                    if filtered:
                        model.load_state_dict(filtered, strict=False)
                        print(f"Loaded partial weights from '{ckpt_path}' ({len(filtered)} keys matched).")
                    else:
                        print(f"Checkpoint '{ckpt_path}' found but no keys matched — starting fresh.")
                break
            except Exception as e:
                print(f"Could not load checkpoint '{ckpt_path}': {e} — starting fresh.")
            break

    # ------------------------------------------------------------------ #
    # Step 6: Training loop                                                #
    # ------------------------------------------------------------------ #
    print(f"\nStarting Training: Epoch {start_epoch + 1} -> {NUM_EPOCHS}")
    print(f"Batches per epoch: {len(train_loader)} | Batch size: 64")
    print(f"Estimated time per epoch: ~2-3 min on RTX 3060\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH [{epoch+1}/{NUM_EPOCHS}]")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mse, val_cos = evaluate_epoch(model, test_loader, device)

        scheduler.step()

        print(
            f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary: "
            f"Train Loss={train_loss:.4f} | "
            f"Val MSE={val_mse:.4f} | Val CosSim={val_cos:.4f}"
        )

        # ---- Save checkpoint ---- #
        save_state = {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch":           epoch,
            "best_mse":        best_mse,
        }
        torch.save(save_state, "checkpoints/latest_jscc_model.pth")

        if val_mse < best_mse:
            best_mse = val_mse
            save_state["best_mse"] = best_mse
            torch.save(save_state, "checkpoints/best_jscc_model.pth")
            print(f" -> Best checkpoint saved! (Val MSE={best_mse:.4f})")


if __name__ == "__main__":
    main()
