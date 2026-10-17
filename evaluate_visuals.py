"""
evaluate_visuals.py
-------------------
Image-space evaluation of trained Semantic Deep JSCC models against digital baselines.

Datasets (all 128x128 frames, the system's input size):
  kodak  Kodak 24 images, downscaled by --kodak-scale and split into 128x128 tiles
  cifar  official CIFAR-10 test split (never seen in training), upsampled to 128x128
  dir    any folder of photos / webcam frames (--image-dir), center-cropped to square

Methods per (channel, SNR):
  every --checkpoints model (its config, e.g. K and receiver mode, is read from the checkpoint)
  digital baselines with the same K as the first checkpoint:
    digital_ideal       WebP over a capacity-achieving code, rate adapted to the SNR
    digital_practical   WebP over discrete MCS rates with a 2 dB gap, adapted to the SNR
    digital_practical@D practical rate fixed for design SNR D (the cliff effect)
  vae_only (no channel): the ceiling of any latent-space scheme

Outputs in --out-dir: metrics.csv, psnr_<dataset>.png, grid_<dataset>.png, fps in the log.

Usage:
  python evaluate_visuals.py --checkpoints checkpoints/k1024_best.pth checkpoints/best_jscc_model.pth
"""

import os
import csv
import time
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

from models.jscc_system import build_from_checkpoint
from baselines.digital import digital_transmit

IMG_SIZE = 128
GRID_SNRS = [0, 10, 20]
DEFAULT_KODAK_DIR = os.path.expanduser('~/datasets/kodak')

to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ------------------------------------------------------------------ datasets
def load_kodak(kodak_dir, scale):
    tiles = []
    for name in sorted(f for f in os.listdir(kodak_dir) if f.lower().endswith('.png')):
        img = Image.open(os.path.join(kodak_dir, name)).convert('RGB')
        img = img.resize((round(img.width / scale), round(img.height / scale)), Image.BICUBIC)
        t = transforms.functional.to_tensor(img)
        for top in range(0, t.shape[1] - IMG_SIZE + 1, IMG_SIZE):
            for left in range(0, t.shape[2] - IMG_SIZE + 1, IMG_SIZE):
                tiles.append(t[:, top:top + IMG_SIZE, left:left + IMG_SIZE])
    return torch.stack(tiles)


def load_cifar_test(num_images, root='./data'):
    dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=to_tensor)
    return torch.stack([dataset[i][0] for i in range(min(num_images, len(dataset)))])


def load_image_dir(path):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
    if not files:
        raise ValueError(f"No images found in {path}")
    imgs = []
    for f in files:
        img = Image.open(os.path.join(path, f)).convert('RGB')
        imgs.append(to_tensor(transforms.CenterCrop(min(img.size))(img)))
    return torch.stack(imgs)


# ------------------------------------------------------------------ metrics
def psnr_per_image(x_hat, x):
    mse = F.mse_loss(x_hat, x, reduction='none').flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def metrics(x_hat, x):
    return psnr_per_image(x_hat, x).mean().item(), ssim(x_hat, x, data_range=1.0, size_average=True).item()


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def run_jscc(models, vae, images, channels, snrs, device, batch_size):
    """Returns ({(method, channel, snr): reconstructions (N,3,H,W) on CPU}, vae_only recon)."""
    recon = {}
    vae_only = []
    latents = [vae.encode(images[i:i + batch_size].to(device)) for i in range(0, len(images), batch_size)]
    for lat in latents:
        vae_only.append(vae.decode(lat).cpu())
    for tag, model in models.items():
        for ch in channels:
            for snr in snrs:
                torch.manual_seed(1234)
                outs = [vae.decode(model.forward_latent(lat, snr_db=float(snr), channel_type=ch)).cpu()
                        for lat in latents]
                recon[(tag, ch, snr)] = torch.cat(outs)
                print(f"   {tag:>22s} {ch:8s} {snr:>2}dB done", flush=True)
    return recon, torch.cat(vae_only)


def run_digital(images, num_symbols, channels, snrs, design_snr):
    recon = {}
    variants = [('digital_ideal', 'ideal', None),
                ('digital_practical', 'practical', None),
                (f'digital_practical@{design_snr}dB', 'practical', design_snr)]
    for name, mode, design in variants:
        for ch in channels:
            for snr in snrs:
                recon[(name, ch, snr)] = torch.stack([
                    digital_transmit(img, num_symbols, snr, ch, design_snr_db=design, mode=mode)[0]
                    for img in images])
        print(f"   {name} done", flush=True)
    return recon


@torch.no_grad()
def measure_fps(model, vae, image, device, n=20):
    x = image.unsqueeze(0).to(device)
    for _ in range(3):
        vae.decode(model.forward_latent(vae.encode(x), 10.0, 'rayleigh'))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        vae.decode(model.forward_latent(vae.encode(x), 10.0, 'rayleigh'))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return n / (time.time() - t)


# ------------------------------------------------------------------ plots
def save_curves(rows, dataset, channels, path):
    fig, axes = plt.subplots(1, len(channels), figsize=(5.2 * len(channels), 4.4), sharey=True)
    axes = [axes] if len(channels) == 1 else axes
    methods = [m for m in dict.fromkeys(r['method'] for r in rows) if m != 'vae_only']
    ceiling = next((r['psnr'] for r in rows if r['method'] == 'vae_only'), None)
    for ax, ch in zip(axes, channels):
        for m in methods:
            pts = sorted((r['snr_db'], r['psnr']) for r in rows if r['method'] == m and r['channel'] == ch)
            style = '--' if m.startswith('digital') else '-'
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, marker='o', markersize=4, label=m)
        if ceiling is not None:
            ax.axhline(ceiling, color='gray', linestyle=':', linewidth=1, label='VAE-only ceiling')
        ax.set_title(ch.upper())
        ax.set_xlabel('SNR (dB)')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('PSNR (dB)')
    axes[-1].legend(fontsize=7, loc='lower right')
    fig.suptitle(f'PSNR vs SNR ({dataset})')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_grid(images, vae_only, recon, primary, digital_name, channel, snrs, path, n=6):
    grid_snrs = [s for s in GRID_SNRS if s in snrs] or snrs
    rows = [('Original', images[:n]), ('VAE only', vae_only[:n])]
    rows += [(f'{primary} {channel} {s}dB', recon[(primary, channel, s)][:n]) for s in grid_snrs]
    rows += [(f'{digital_name} {channel} {s}dB', recon[(digital_name, channel, s)][:n]) for s in grid_snrs]
    img = make_grid(torch.cat([r[1] for r in rows]), nrow=n, padding=2).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(n * 1.6 + 3.2, len(rows) * 1.6))
    ax.imshow(img)
    ax.set_yticks([(IMG_SIZE + 2) * (i + 0.5) for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', default=['checkpoints/best_jscc_model.pth'])
    parser.add_argument('--dataset', choices=['kodak', 'cifar', 'dir'], default='kodak')
    parser.add_argument('--kodak-dir', default=DEFAULT_KODAK_DIR)
    parser.add_argument('--kodak-scale', type=float, default=2.0)
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--num-images', type=int, default=200, help='CIFAR only')
    parser.add_argument('--channels', nargs='+', default=['awgn', 'rician', 'rayleigh'])
    parser.add_argument('--snrs', nargs='+', type=int, default=[0, 5, 10, 15, 20])
    parser.add_argument('--design-snr', type=int, default=10)
    parser.add_argument('--no-digital', action='store_true')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.dataset == 'kodak':
        images = load_kodak(args.kodak_dir, args.kodak_scale)
    elif args.dataset == 'cifar':
        images = load_cifar_test(args.num_images)
    else:
        images = load_image_dir(args.image_dir)
    print(f"Dataset '{args.dataset}': {len(images)} frames of 128x128")

    models, vae = {}, None
    for path in args.checkpoints:
        model, epoch = build_from_checkpoint(path, device, use_vae=vae is None)
        if vae is None:
            vae = model.vae
        tag = os.path.splitext(os.path.basename(path))[0].replace('_best', '').replace('best_jscc_model', 'k256')
        models[tag] = model
        print(f"Loaded {tag}: {model.config} (epoch {epoch + 1})")
    primary = next(iter(models))
    num_symbols = models[primary].config['num_symbols']

    print("\nRunning JSCC models...")
    recon, vae_only = run_jscc(models, vae, images, args.channels, args.snrs, device, args.batch_size)
    digital_name = None
    if not args.no_digital:
        print(f"\nRunning digital baselines (K={num_symbols}, CPU)...")
        recon.update(run_digital(images, num_symbols, args.channels, args.snrs, args.design_snr))
        digital_name = 'digital_practical'

    rows = []
    p, s = metrics(vae_only, images)
    rows.append({'dataset': args.dataset, 'method': 'vae_only', 'num_symbols': '', 'channel': '',
                 'snr_db': '', 'psnr': p, 'ssim': s})
    for (method, ch, snr), x_hat in recon.items():
        p, s = metrics(x_hat, images)
        k = models[method].config['num_symbols'] if method in models else num_symbols
        rows.append({'dataset': args.dataset, 'method': method, 'num_symbols': k, 'channel': ch,
                     'snr_db': snr, 'psnr': p, 'ssim': s})

    with open(os.path.join(args.out_dir, f'metrics_{args.dataset}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'method':>28s} {'channel':8s} " + ' '.join(f'{s:>5}dB' for s in args.snrs))
    print(f"{'vae_only':>28s} {'-':8s} PSNR {rows[0]['psnr']:.2f}")
    for method in dict.fromkeys(r['method'] for r in rows[1:]):
        for ch in args.channels:
            vals = {r['snr_db']: r['psnr'] for r in rows if r['method'] == method and r['channel'] == ch}
            print(f"{method:>28s} {ch:8s} " + ' '.join(f'{vals[s]:7.2f}' for s in args.snrs))

    save_curves(rows, args.dataset, args.channels, os.path.join(args.out_dir, f'psnr_{args.dataset}.png'))
    if digital_name:
        save_grid(images, vae_only, recon, primary, digital_name, args.channels[-1], args.snrs,
                  os.path.join(args.out_dir, f'grid_{args.dataset}.png'))

    print("\nEnd-to-end speed (batch 1, encode + JSCC + decode):")
    for tag, model in models.items():
        print(f"   {tag}: {measure_fps(model, vae, images[0], device):.1f} fps")
    print(f"\nSaved results to '{args.out_dir}/'")


if __name__ == '__main__':
    main()
