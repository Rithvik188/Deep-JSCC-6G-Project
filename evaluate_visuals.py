"""
evaluate_visuals.py
-------------------
Image-space evaluation of the trained Semantic Deep JSCC model.

Uses the official CIFAR-10 test split (never seen in training: the training latents
were carved out of the CIFAR-10 train split), resized to 128x128 like training.
Optionally also evaluates a folder of real photos / webcam captures (--image-dir).

Outputs (in results/):
  metrics.csv            PSNR / SSIM / latent MSE per (dataset, channel, SNR)
  psnr_vs_snr.png        PSNR vs SNR for each channel + VAE-only ceiling
  grid_<set>.png         Originals, VAE-only, and reconstructions per channel/SNR

Usage:
  python evaluate_visuals.py
  python evaluate_visuals.py --num-images 2000 --image-dir my_webcam_frames
"""

import os
import csv
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid
from pytorch_msssim import ssim

from models.jscc_system import DeepJSCCSystem, load_jscc_weights

CHANNELS = ['awgn', 'rician', 'rayleigh']
SNR_LIST = [0, 5, 10, 15, 20]
GRID_SNRS = [0, 10, 20]
IMG_SIZE = 128

to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_cifar_test(num_images, root='./data'):
    dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=to_tensor)
    n = min(num_images, len(dataset))
    return torch.stack([dataset[i][0] for i in range(n)])


def load_image_dir(path):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
    if not files:
        raise ValueError(f"No images found in {path}")
    # Center-crop to square first so faces/scenes are not squashed
    imgs = []
    for f in files:
        img = Image.open(os.path.join(path, f)).convert('RGB')
        side = min(img.size)
        imgs.append(to_tensor(transforms.CenterCrop(side)(img)))
    return torch.stack(imgs)


def psnr_per_image(x_hat, x):
    mse = F.mse_loss(x_hat, x, reduction='none').flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


@torch.no_grad()
def evaluate_set(model, images, device, batch_size=50):
    """Returns (rows, grid_tensor). Rows are dicts with averaged metrics per condition."""
    model.eval()
    rows = []
    sums = {}  # key -> [psnr_sum, ssim_sum, latent_mse_sum]
    grid_n = min(8, len(images))
    grid = {}

    def add(key, x_hat, x, lat_mse):
        s = sums.setdefault(key, [0.0, 0.0, 0.0])
        s[0] += psnr_per_image(x_hat, x).sum().item()
        s[1] += ssim(x_hat, x, data_range=1.0, size_average=False).sum().item()
        s[2] += lat_mse * x.size(0)

    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size].to(device)
        latents = model.vae.encode(x)
        first = start == 0

        x_vae = model.vae.decode(latents)
        add(('vae_only', None), x_vae, x, 0.0)
        if first:
            grid['original'] = x[:grid_n].cpu()
            grid['vae_only'] = x_vae[:grid_n].cpu()

        for ch in CHANNELS:
            for snr in SNR_LIST:
                latents_hat = model.forward_latent(latents, snr_db=float(snr), channel_type=ch)
                x_hat = model.vae.decode(latents_hat)
                add((ch, snr), x_hat, x, F.mse_loss(latents_hat, latents).item())
                if first and snr in GRID_SNRS:
                    grid[(ch, snr)] = x_hat[:grid_n].cpu()

    n = len(images)
    for (ch, snr), (p, s, m) in sums.items():
        rows.append({'channel': ch, 'snr_db': '' if snr is None else snr,
                     'psnr': p / n, 'ssim': s / n, 'latent_mse': m / n})
    return rows, grid


def save_grid(grid, path, title):
    order = ['original', 'vae_only'] + [(ch, snr) for ch in CHANNELS for snr in GRID_SNRS]
    labels = ['Original', 'VAE only'] + [f"{ch} {snr}dB" for ch in CHANNELS for snr in GRID_SNRS]
    nrow = grid['original'].size(0)
    tiles = torch.cat([grid[k] for k in order], dim=0)
    img = make_grid(tiles, nrow=nrow, padding=2).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(nrow * 1.6 + 1.5, len(order) * 1.6))
    ax.imshow(img)
    ax.set_yticks([(IMG_SIZE + 2) * (i + 0.5) for i in range(len(order))])
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_curve(all_rows, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'awgn': 'tab:blue', 'rician': 'tab:green', 'rayleigh': 'tab:red'}
    styles = {'cifar_test': '-', 'custom': '--'}
    for dset, rows in all_rows.items():
        for ch in CHANNELS:
            pts = [(r['snr_db'], r['psnr']) for r in rows if r['channel'] == ch]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], styles.get(dset, '-'),
                    marker='o', color=colors[ch], label=f"{ch} ({dset})")
        ceiling = next(r['psnr'] for r in rows if r['channel'] == 'vae_only')
        ax.axhline(ceiling, color='gray', linestyle=styles.get(dset, '-'), linewidth=1,
                   label=f"VAE-only ceiling ({dset})")
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Semantic Deep JSCC: PSNR vs SNR')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_jscc_model.pth')
    parser.add_argument('--num-images', type=int, default=1000)
    parser.add_argument('--image-dir', default=None, help='Optional folder of real photos/webcam frames')
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DeepJSCCSystem(use_vae=True).to(device)
    epoch = load_jscc_weights(model, args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch index {epoch})")

    sets = {'cifar_test': load_cifar_test(args.num_images)}
    if args.image_dir:
        sets['custom'] = load_image_dir(args.image_dir)

    all_rows = {}
    with open(os.path.join(args.out_dir, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'channel', 'snr_db', 'psnr', 'ssim', 'latent_mse'])
        writer.writeheader()
        for name, images in sets.items():
            print(f"\nEvaluating '{name}' ({len(images)} images)...")
            rows, grid = evaluate_set(model, images, device)
            all_rows[name] = rows
            for r in rows:
                writer.writerow({'dataset': name, **r})
                snr = 'n/a' if r['snr_db'] == '' else f"{r['snr_db']:>2}dB"
                print(f"  {r['channel']:9s} {snr:>5s} | PSNR {r['psnr']:6.2f} dB | "
                      f"SSIM {r['ssim']:.3f} | latent MSE {r['latent_mse']:.3f}")
            save_grid(grid, os.path.join(args.out_dir, f'grid_{name}.png'), f"Reconstructions ({name})")

    save_curve(all_rows, os.path.join(args.out_dir, 'psnr_vs_snr.png'))
    print(f"\nSaved metrics.csv, psnr_vs_snr.png and grids to '{args.out_dir}/'")


if __name__ == '__main__':
    main()
