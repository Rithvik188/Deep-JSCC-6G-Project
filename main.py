"""
main.py
-------
Entry point for Deep JSCC 6G training.

Step 1 (auto): If cached latents don't exist, runs precompute_latents.py first.
Step 2:        Trains only ViT Transmitter + Channel + ViT Receiver on cached latents.

Training is resumable: a full checkpoint (model, optimizer, scheduler, epoch) is written to
checkpoints/<tag>_latest.pth after every epoch, and re-running the same command continues from it.
The best epoch (lowest validation MSE averaged over AWGN/Rician/Rayleigh at 10 dB) is also
copied to checkpoints/<tag>_best.pth, which server.py and evaluate_visuals.py can load.

Examples:
  python main.py                                   # K=256, MMSE receiver, CIFAR latents
  python main.py --symbols 1024 --tag k1024
  python main.py --receiver raw --tag k256_raw     # raw-CSI ablation
  python main.py --symbols 1024 --latents div2k --tag k1024_div2k
"""

import os
import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.latent_dataset import get_latent_dataloaders
from models.jscc_system import DeepJSCCSystem, MODEL_VERSION, DEFAULT_CONFIG
from train import train_epoch, evaluate_epoch

BATCH_SIZE = 64


def latent_paths(source):
    """Cached latent files for a dataset source ('cifar' keeps the original file names)."""
    prefix = "latents" if source == "cifar" else f"latents_{source}"
    return f"./data/{prefix}_train.pt", f"./data/{prefix}_test.pt"


def maybe_precompute(source):
    """Auto-run precompute if cached latents are missing."""
    train_path, test_path = latent_paths(source)
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("=" * 60)
        print(f"Cached '{source}' latents not found. Running one-time precomputation.")
        print("=" * 60)
        from precompute_latents import precompute_and_save
        precompute_and_save(source)


def try_resume(model, optimizer, scheduler, device, latest_ckpt):
    """Resumes from the latest checkpoint. Returns (start_epoch, best_mse)."""
    if not os.path.exists(latest_ckpt):
        return 0, float("inf")

    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=True)
    version = ckpt.get("model_version", 1) if isinstance(ckpt, dict) else 1
    if version != MODEL_VERSION or "scheduler_state" not in ckpt:
        raise RuntimeError(
            f"'{latest_ckpt}' is from an older model version ({version}) and cannot be resumed. "
            f"Move it out of checkpoints/ to start a fresh run."
        )
    config = {**DEFAULT_CONFIG, **ckpt.get("config", {})}
    if config != model.config:
        raise RuntimeError(f"'{latest_ckpt}' is a {config} run, but this command trains {model.config}. "
                           f"Use a different --tag.")

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from '{latest_ckpt}' -> continuing from Epoch {start_epoch + 1}")
    return start_epoch, ckpt["best_mse"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=DEFAULT_CONFIG["num_symbols"],
                        help="Complex channel symbols per 128x128 frame (bandwidth)")
    parser.add_argument("--receiver", choices=["mmse", "raw"], default=DEFAULT_CONFIG["receiver_mode"])
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_CONFIG["embed_dim"])
    parser.add_argument("--depth", type=int, default=DEFAULT_CONFIG["depth"])
    parser.add_argument("--latents", default="cifar", help="Cached latent source: cifar | div2k")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tag", default=None,
                        help="Checkpoint name prefix (default: the original best/latest_jscc_model names)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tag:
        latest_ckpt = f"checkpoints/{args.tag}_latest.pth"
        best_ckpt = f"checkpoints/{args.tag}_best.pth"
    else:
        latest_ckpt = "checkpoints/latest_jscc_model.pth"
        best_ckpt = "checkpoints/best_jscc_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    maybe_precompute(args.latents)

    print("Loading cached latent dataset...")
    train_path, test_path = latent_paths(args.latents)
    train_loader, test_loader = get_latent_dataloaders(
        train_path=train_path,
        test_path=test_path,
        batch_size=BATCH_SIZE,
    )

    # Transmitter + Channel + Receiver only; the VAE is used by server.py at inference
    model = DeepJSCCSystem(use_vae=False, num_symbols=args.symbols, receiver_mode=args.receiver,
                           embed_dim=args.embed_dim, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model config: {model.config} | {n_params/1e6:.1f}M params | latents: {args.latents}")

    num_epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    start_epoch, best_mse = try_resume(model, optimizer, scheduler, device, latest_ckpt)

    print(f"\nStarting Training: Epoch {start_epoch + 1} -> {num_epochs}")
    print(f"Batches per epoch: {len(train_loader)} | Batch size: {BATCH_SIZE}\n")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH [{epoch+1}/{num_epochs}]  (lr={scheduler.get_last_lr()[0]:.2e})")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val = evaluate_epoch(model, test_loader, device)
        val_mse = sum(mse for mse, _ in val.values()) / len(val)

        scheduler.step()

        per_channel = " | ".join(f"{ch} {mse:.4f}" for ch, (mse, _) in val.items())
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary: Train Loss={train_loss:.4f} | "
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
            "config":          model.config,
            "latents":         args.latents,
        }
        torch.save(save_state, latest_ckpt)
        if is_best:
            torch.save(save_state, best_ckpt)
            print(f" -> Best checkpoint saved! (mean Val MSE={best_mse:.4f})")


if __name__ == "__main__":
    main()
