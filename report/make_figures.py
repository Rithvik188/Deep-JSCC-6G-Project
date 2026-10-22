"""
report/make_figures.py
----------------------
Presentation figures for the project update / report.

Plots read results/kodak/metrics_kodak.csv (produced by evaluate_visuals.py); only the visual
comparison grid runs the models itself (a handful of images).

  fig1_system.png        block diagram of the pipeline
  fig2_equalization.png  raw-CSI vs MMSE-equalized receiver on Rayleigh fading (ablation)
  fig3_bandwidth.png     PSNR vs SNR for K = 256 / 512 / 1024 symbols
  fig4_vs_digital.png    Deep JSCC vs digital baselines, one panel per channel
  fig5_visual.png        reconstructions: JSCC vs practical digital at 20 / 10 / 0 dB (Rayleigh)

Usage:
  python report/make_figures.py [--metrics results/kodak/metrics_kodak.csv] [--out report/figures]
  python report/make_figures.py --skip-visual     # plots only, no GPU needed
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------ style (light, slide use)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'      # categorical slots 1-3 (validated)
RAMP = {256: '#86b6ef', 512: '#2a78d6', 1024: '#104281'}   # ordinal blue ramp (validated)
REF_GRAY = '#898781'

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'font.size': 11,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.linewidth': 1.0,
    'axes.labelcolor': INK_2,
    'axes.titlecolor': INK,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'grid.linestyle': '-',
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelcolor': MUTED,
    'ytick.labelcolor': MUTED,
    'legend.frameon': False,
    'legend.labelcolor': INK_2,
    'lines.linewidth': 2.0,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
})
LINE = dict(linewidth=2.0, marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)


def title(fig, main, sub, y_main=0.975, y_sub=0.915):
    fig.text(0.012, y_main, main, ha='left', va='top', fontsize=15, fontweight='semibold', color=INK)
    fig.text(0.012, y_sub, sub, ha='left', va='top', fontsize=10.5, color=INK_2)


def end_label(ax, xs, ys, text, color_dot, dy=0.0):
    """Direct label at a line's right end: a colored dot + text in ink (never series-colored text)."""
    ax.annotate(text, xy=(xs[-1], ys[-1]), xytext=(8, dy), textcoords='offset points',
                va='center', ha='left', fontsize=9.5, color=INK_2, annotation_clip=False)


def load_metrics(path):
    data = defaultdict(dict)  # (method, channel) -> {snr: psnr}
    ceiling = None
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['method'] == 'vae_only':
                ceiling = float(r['psnr'])
                continue
            data[(r['method'], r['channel'])][int(r['snr_db'])] = float(r['psnr'])
    return data, ceiling


def series(data, method, channel):
    pts = sorted(data[(method, channel)].items())
    return [p[0] for p in pts], [p[1] for p in pts]


def finish_axes(ax, ylim):
    ax.set_xlabel('Channel SNR (dB)')
    ax.set_xlim(-0.5, 20.5)
    ax.set_xticks(range(0, 21, 5))
    ax.set_ylim(*ylim)
    ax.tick_params(length=0)


# ------------------------------------------------------------------ fig 1: system diagram
def fig_system(out):
    fig = plt.figure(figsize=(12, 3.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86])
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.9)
    blocks = [
        (0.15, 'Camera\nframe', '128x128 RGB', False),
        (2.12, 'VAE\nencoder', 'frozen SD-VAE\n-> 4x16x16 latent', False),
        (4.09, 'ViT\ntransmitter', '-> K complex\nsymbols, unit power', True),
        (6.06, 'Wireless\nchannel', 'y = h·x + n\nAWGN, Rician,\nor Rayleigh', False),
        (8.03, 'MMSE equalizer\n+ ViT receiver', 'uses CSI h\nand SNR', True),
        (10.0, 'VAE\ndecoder', 'frozen\n-> 128x128 RGB', False),
    ]
    w, h, y = 1.8, 1.95, 0.7
    for x, name, detail, trained in blocks:
        face = '#e8f1fc' if trained else '#f0efec'
        edge = BLUE if trained else AXIS
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    facecolor=face, edgecolor=edge, linewidth=1.2))
        ax.text(x + w / 2, y + h - 0.45, name, ha='center', va='center', fontsize=10.5,
                fontweight='semibold', color=INK, linespacing=1.15)
        ax.text(x + w / 2, y + 0.55, detail, ha='center', va='center', fontsize=8.8, color=INK_2,
                linespacing=1.3)
    for i in range(len(blocks) - 1):
        x0 = blocks[i][0] + w
        ax.annotate('', xy=(blocks[i + 1][0], y + h / 2), xytext=(x0, y + h / 2),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4, shrinkA=2, shrinkB=2))
    ax.text(0.15, 0.25, 'Blue = trained end-to-end over random channel types and SNRs (0-20 dB). '
            'Grey = frozen pretrained Stable Diffusion VAE.', fontsize=9.5, color=INK_2)
    fig.text(0.012, 0.96, 'Semantic Deep JSCC pipeline', ha='left', va='top', fontsize=15,
             fontweight='semibold', color=INK)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_system_compact(out):
    """Two-row (snake) version of the pipeline for portrait documents: row 1 runs left to right,
    row 2 right to left, so the text stays legible at page width."""
    fig = plt.figure(figsize=(7.5, 4.3))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 4.3)
    w, h = 2.1, 1.55
    xs = [0.15, 2.7, 5.25]
    y_top, y_bot = 2.55, 0.25
    blocks = [  # (x, y, name, detail, trained) in signal order
        (xs[0], y_top, 'Camera frame', '128x128 RGB', False),
        (xs[1], y_top, 'VAE encoder', 'frozen SD-VAE\n-> 4x16x16 latent', False),
        (xs[2], y_top, 'ViT transmitter', '-> K complex symbols,\nunit average power', True),
        (xs[2], y_bot, 'Wireless channel', 'y = h·x + n\nAWGN, Rician or Rayleigh', False),
        (xs[1], y_bot, 'MMSE equalizer\n+ ViT receiver', 'uses CSI h and SNR', True),
        (xs[0], y_bot, 'VAE decoder', 'frozen\n-> 128x128 RGB', False),
    ]
    for x, y, name, detail, trained in blocks:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
                                    facecolor='#e8f1fc' if trained else '#f0efec',
                                    edgecolor=BLUE if trained else AXIS, linewidth=1.2))
        ax.text(x + w / 2, y + h - 0.42, name, ha='center', va='center', fontsize=11,
                fontweight='semibold', color=INK, linespacing=1.1)
        ax.text(x + w / 2, y + 0.42, detail, ha='center', va='center', fontsize=9.2, color=INK_2,
                linespacing=1.3)
    arrow = dict(arrowstyle='-|>', color=MUTED, lw=1.4, shrinkA=2, shrinkB=2)
    for i in range(len(blocks) - 1):
        (x0, y0), (x1, y1) = blocks[i][:2], blocks[i + 1][:2]
        if y0 == y1 and x1 > x0:
            ax.annotate('', xy=(x1, y1 + h / 2), xytext=(x0 + w, y0 + h / 2), arrowprops=arrow)
        elif y0 == y1:
            ax.annotate('', xy=(x1 + w, y1 + h / 2), xytext=(x0, y0 + h / 2), arrowprops=arrow)
        else:
            ax.annotate('', xy=(x1 + w / 2, y1 + h), xytext=(x0 + w / 2, y0), arrowprops=arrow)
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 2: equalization ablation
def fig_equalization(data, ceiling, out, titled=True):
    fig, ax = plt.subplots(figsize=(8, 4.8 if titled else 4.2))
    fig.subplots_adjust(top=0.80 if titled else 0.97, right=0.80, left=0.09, bottom=0.13)
    xs, ys = series(data, 'k256', 'rayleigh')
    ax.plot(xs, ys, color=BLUE, label='MMSE equalization before the ViT (ours)', **LINE)
    end_label(ax, xs, ys, 'MMSE-equalized', BLUE)
    xr, yr = series(data, 'k256_raw', 'rayleigh')
    ax.plot(xr, yr, color=ORANGE, label='Raw [y, h] into the ViT', **LINE)
    end_label(ax, xr, yr, 'Raw CSI input', ORANGE)
    finish_axes(ax, (min(yr + ys) - 1.5, max(ys) + 1.5))
    ax.set_ylabel('PSNR (dB)')
    ax.legend(loc='upper left', fontsize=9.5)
    if titled:
        title(fig, 'Explicit equalization makes Rayleigh fading learnable',
              'Rayleigh fading, K = 256 symbols, Kodak test images. Same model, data and training; '
              'only the receiver input differs.')
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ fig 3: bandwidth
def fig_bandwidth(data, ceiling, out, channel='rayleigh', titled=True):
    fig, ax = plt.subplots(figsize=(8, 4.8 if titled else 4.2))
    fig.subplots_adjust(top=0.80 if titled else 0.97, right=0.80, left=0.09, bottom=0.13)
    tags = {256: 'k256', 512: 'k512', 1024: 'k1024'}
    all_y = []
    for k, tag in tags.items():
        xs, ys = series(data, tag, channel)
        all_y += ys
        ratio = 128 * 128 * 3 // k
        ax.plot(xs, ys, color=RAMP[k], label=f'K = {k} symbols (1/{ratio} of pixel values)', **LINE)
        end_label(ax, xs, ys, f'K = {k}', RAMP[k])
    ax.axhline(ceiling, color=REF_GRAY, linewidth=1.2, linestyle=(0, (2, 2)))
    ax.annotate('VAE-only ceiling (no channel)', xy=(0, ceiling), xytext=(2, 5), textcoords='offset points',
                fontsize=9, color=MUTED)
    finish_axes(ax, (min(all_y) - 1.5, ceiling + 1.5))
    ax.set_ylabel('PSNR (dB)')
    ax.legend(loc='lower right', fontsize=9.5)
    if titled:
        title(fig, 'More channel symbols per frame, sharper reconstructions',
              f'{channel.capitalize()} fading, Kodak test images, models trained on CIFAR-10 latents.')
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ fig 4: vs digital
def fig_vs_digital(data, ceiling, out, jscc='k1024_div2k', design='10dB'):
    channels = ['awgn', 'rician', 'rayleigh']
    names = {'awgn': 'AWGN', 'rician': 'Rician (LoS)', 'rayleigh': 'Rayleigh (NLoS)'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), sharey=True)
    fig.subplots_adjust(top=0.72, left=0.06, right=0.99, bottom=0.14, wspace=0.08)
    all_y = []
    for ax, ch in zip(axes, channels):
        for method, color, label, kw in [
            ('digital_ideal', REF_GRAY, 'Digital, ideal capacity-achieving code (upper bound)',
             dict(linewidth=1.3, marker=None)),
            # Practical digital rates change in discrete steps: draw steps, not slopes
            (f'digital_practical@{design}', AQUA, f'Digital, practical code, rate fixed for {design[:-2]} dB',
             dict(drawstyle='steps-post')),
            ('digital_practical', ORANGE, 'Digital, practical code, rate adapted to SNR',
             dict(drawstyle='steps-post')),
            (jscc, BLUE, 'Deep JSCC (ours)', {}),
        ]:
            xs, ys = series(data, method, ch)
            all_y += ys
            style = {**LINE, **kw}
            ax.plot(xs, ys, color=color, label=label, **style)
        ax.axhline(ceiling, color=REF_GRAY, linewidth=1.0, linestyle=(0, (2, 2)))
        ax.set_title(names[ch], loc='left', fontsize=11.5, color=INK, pad=6)
        finish_axes(ax, (min(all_y) - 1.0, max(all_y + [ceiling]) + 1.0))
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].annotate('VAE ceiling', xy=(20, ceiling), xytext=(-2, 4), textcoords='offset points',
                     ha='right', fontsize=8.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [3, 2, 1, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left',
               bbox_to_anchor=(0.055, 0.86), ncol=2, fontsize=9.5)
    title(fig, 'Deep JSCC degrades gracefully; digital wins at high SNR but falls off a cliff below its design point',
          'Same bandwidth for every scheme: K = 1024 complex channel symbols per 128x128 frame. '
          'Kodak test images. Failed digital frames count as flat grey. Dotted line: VAE ceiling.')
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ fig 5: visual comparison
def fig_visual(out, checkpoint, extra_image=None, channel='rayleigh', snrs=(20, 10, 0)):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from models.jscc_system import build_from_checkpoint
    from baselines.digital import digital_transmit

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = build_from_checkpoint(checkpoint, device, use_vae=True)
    k = model.config['num_symbols']
    tf = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    kodak = os.path.expanduser('~/datasets/kodak')
    pil = [Image.open(os.path.join(kodak, f'kodim{n}.png')).convert('RGB') for n in ('04', '23', '15')]
    if extra_image and os.path.exists(extra_image):
        pil.append(Image.open(extra_image).convert('RGB'))
    x = torch.stack([tf(T.CenterCrop(min(im.size))(im)) for im in pil]).to(device)

    cols = [('Original', None)] + [(f'Deep JSCC\n{s} dB', ('jscc', s)) for s in snrs] + \
           [(f'Digital\n{s} dB', ('digital', s)) for s in snrs]
    tiles, failed = {}, set()
    with torch.no_grad():
        lat = model.vae.encode(x)
        for s in snrs:
            torch.manual_seed(0)
            tiles[('jscc', s)] = model.vae.decode(model.forward_latent(lat, float(s), channel)).cpu()
            dig = []
            for i in range(len(x)):
                img, nbytes = digital_transmit(x[i].cpu(), k, s, channel, mode='practical')
                dig.append(img)
                if nbytes == 0:
                    failed.add((s, i))
            tiles[('digital', s)] = torch.stack(dig)

    n_rows, n_cols = len(x), len(cols)
    height = 1.75 * n_rows + 1.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.75 * n_cols, height))
    fig.subplots_adjust(top=1 - 1.5 / height, bottom=0.01, left=0.01, right=0.99,
                        wspace=0.04, hspace=0.04)
    for c, (label, key) in enumerate(cols):
        for r in range(n_rows):
            ax = axes[r, c]
            img = x[r].cpu() if key is None else tiles[key][r]
            ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if key and key[0] == 'digital' and (key[1], r) in failed:
                ax.text(0.5, 0.5, 'decoding\nfailed', transform=ax.transAxes, ha='center', va='center',
                        fontsize=9, color=INK_2)
            if r == 0:
                ax.set_title(label, fontsize=9.5, color=INK, pad=4)
    title(fig, f'Reconstructions over {channel.capitalize()} fading (K = {k} symbols)',
          'Digital = WebP + practical channel code with rate adapted to the SNR, same bandwidth. '
          'Bottom row: a real webcam frame from the live demo.',
          y_main=1 - 0.15 / height, y_sub=1 - 0.55 / height)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='results/kodak/metrics_kodak.csv')
    parser.add_argument('--out', default='report/figures')
    parser.add_argument('--checkpoint', default='checkpoints/k1024_div2k_best.pth')
    parser.add_argument('--extra-image', default=None, help='Optional webcam frame for the visual grid')
    parser.add_argument('--skip-visual', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig_system(os.path.join(args.out, 'fig1_system.png'))
    data, ceiling = load_metrics(args.metrics)
    fig_equalization(data, ceiling, os.path.join(args.out, 'fig2_equalization.png'))
    fig_bandwidth(data, ceiling, os.path.join(args.out, 'fig3_bandwidth.png'))
    # Document versions: no in-chart titles (the document's captions carry them), compact diagram
    doc_dir = os.path.join(args.out, 'doc')
    os.makedirs(doc_dir, exist_ok=True)
    fig_system_compact(os.path.join(doc_dir, 'fig1_system.png'))
    fig_equalization(data, ceiling, os.path.join(doc_dir, 'fig2_equalization.png'), titled=False)
    fig_bandwidth(data, ceiling, os.path.join(doc_dir, 'fig3_bandwidth.png'), titled=False)
    fig_vs_digital(data, ceiling, os.path.join(args.out, 'fig4_vs_digital.png'))
    if not args.skip_visual:
        fig_visual(os.path.join(args.out, 'fig5_visual.png'), args.checkpoint, args.extra_image)
    print(f"Figures written to {args.out}/")


if __name__ == '__main__':
    main()
