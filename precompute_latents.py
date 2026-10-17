"""
precompute_latents.py
---------------------
ONE-TIME script. Encodes a training image set through the frozen Stable Diffusion VAE and
saves the resulting latent tensors to disk, so training never runs the VAE.

Sources:
  cifar  -> data/latents_train.pt, data/latents_test.pt
            CIFAR-10 train split (32 px, upsampled to 128), random 80/20 split.
  div2k  -> data/latents_div2k_train.pt, data/latents_div2k_test.pt
            DIV2K train HR photos. Each image is resized to a random short side in
            [128, 512] px and randomly cropped/flipped to 128x128, so crops range from whole
            scenes to close-ups. Split by IMAGE (no test crop comes from a training image).

Usage:
  python precompute_latents.py --source div2k [--div2k-dir C:/Users/rithv/datasets/div2k/DIV2K_train_HR]
"""

import os
import argparse
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from models.vae import SemanticVAE

LATENT_DIR = "./data"
DEFAULT_DIV2K_DIR = os.path.expanduser("~/datasets/div2k/DIV2K_train_HR")
IMG_SIZE = 128
SEED = 0


def output_paths(source):
    prefix = "latents" if source == "cifar" else f"latents_{source}"
    return (os.path.join(LATENT_DIR, f"{prefix}_train.pt"),
            os.path.join(LATENT_DIR, f"{prefix}_test.pt"))


def random_crops(img, n, rng):
    """n random 128x128 crops of a PIL image after random rescaling and horizontal flips."""
    crops = []
    for _ in range(n):
        short_side = rng.randint(IMG_SIZE, 512)
        scale = short_side / min(img.size)
        resized = img.resize((max(IMG_SIZE, round(img.width * scale)),
                              max(IMG_SIZE, round(img.height * scale))), Image.BICUBIC)
        left = rng.randint(0, resized.width - IMG_SIZE)
        top = rng.randint(0, resized.height - IMG_SIZE)
        crop = resized.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
        if rng.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        crops.append(TF.to_tensor(crop))
    return torch.stack(crops)


@torch.no_grad()
def encode_batches(vae, images, device, batch_size=50):
    return torch.cat([vae.encode(images[i:i + batch_size].to(device)).cpu()
                      for i in range(0, len(images), batch_size)])


def precompute_div2k(vae, device, div2k_dir, crops_per_image=50, test_fraction=0.1):
    files = sorted(f for f in os.listdir(div2k_dir) if f.lower().endswith(".png"))
    if not files:
        raise FileNotFoundError(f"No DIV2K PNGs found in {div2k_dir}")
    rng = random.Random(SEED)
    order = files[:]
    rng.shuffle(order)
    n_test = max(1, int(len(order) * test_fraction))
    splits = {"test": order[:n_test], "train": order[n_test:]}

    train_out, test_out = output_paths("div2k")
    for split, out_path in [("train", train_out), ("test", test_out)]:
        if os.path.exists(out_path):
            print(f"[Precompute] {out_path} already exists - skipping.")
            continue
        latents = []
        names = splits[split]
        print(f"[Precompute] DIV2K {split}: {len(names)} images x {crops_per_image} crops", flush=True)
        for i, name in enumerate(names):
            img = Image.open(os.path.join(div2k_dir, name)).convert("RGB")
            latents.append(encode_batches(vae, random_crops(img, crops_per_image, rng), device))
            if (i + 1) % 20 == 0 or (i + 1) == len(names):
                print(f"   -> Image [{i+1}/{len(names)}]", flush=True)
        latents = torch.cat(latents)
        torch.save(latents, out_path)
        print(f"[Precompute] Saved {split}: shape={tuple(latents.shape)} -> {out_path}", flush=True)


def precompute_cifar(vae, device):
    from data.dataset import get_face_dataloaders
    torch.manual_seed(SEED)
    train_loader, test_loader = get_face_dataloaders(batch_size=64, img_size=IMG_SIZE)
    for loader, out_path in zip((train_loader, test_loader), output_paths("cifar")):
        if os.path.exists(out_path):
            print(f"[Precompute] {out_path} already exists - skipping.")
            continue
        with torch.no_grad():
            latents = torch.cat([vae.encode(images.to(device)).cpu() for images, _ in loader])
        torch.save(latents, out_path)
        print(f"[Precompute] Saved: shape={tuple(latents.shape)} -> {out_path}")


def precompute_and_save(source="cifar", div2k_dir=DEFAULT_DIV2K_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Precompute] Source: {source} | device: {device}")
    vae = SemanticVAE().to(device).eval()
    if source == "div2k":
        precompute_div2k(vae, device, div2k_dir)
    elif source == "cifar":
        precompute_cifar(vae, device)
    else:
        raise ValueError(f"Unknown source: {source}")
    print("[Precompute] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cifar", "div2k"], default="cifar")
    parser.add_argument("--div2k-dir", default=DEFAULT_DIV2K_DIR)
    args = parser.parse_args()
    precompute_and_save(args.source, args.div2k_dir)
