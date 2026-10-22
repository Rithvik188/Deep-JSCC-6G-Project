"""
report/build_update_pdf.py
--------------------------
Builds the project progress-update PDF (report/Project_Update.pdf) from report/figures/.

Usage:  python report/build_update_pdf.py
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                KeepTogether, PageBreak, ListFlowable, ListItem)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures', 'doc')  # untitled document versions from make_figures.py
OUT = os.path.join(HERE, 'Project_Update.pdf')

FONT_DIR = r'C:\Windows\Fonts'
pdfmetrics.registerFont(TTFont('Segoe', os.path.join(FONT_DIR, 'segoeui.ttf')))
pdfmetrics.registerFont(TTFont('Segoe-Bold', os.path.join(FONT_DIR, 'segoeuib.ttf')))
pdfmetrics.registerFont(TTFont('Segoe-Italic', os.path.join(FONT_DIR, 'segoeuii.ttf')))
pdfmetrics.registerFont(TTFont('Segoe-Semi', os.path.join(FONT_DIR, 'seguisb.ttf')))
pdfmetrics.registerFontFamily('Segoe', normal='Segoe', bold='Segoe-Bold', italic='Segoe-Italic',
                              boldItalic='Segoe-Bold')

INK = colors.HexColor('#0b0b0b')
INK_2 = colors.HexColor('#52514e')
MUTED = colors.HexColor('#898781')
RULE = colors.HexColor('#e1e0d9')
ACCENT = colors.HexColor('#2a78d6')
PANEL = colors.HexColor('#f4f3ef')

S = {
    'title': ParagraphStyle('title', fontName='Segoe-Semi', fontSize=22, leading=27, textColor=INK,
                            spaceAfter=4),
    'subtitle': ParagraphStyle('subtitle', fontName='Segoe', fontSize=12, leading=16, textColor=INK_2,
                               spaceAfter=2),
    'meta': ParagraphStyle('meta', fontName='Segoe', fontSize=9.5, leading=13, textColor=MUTED,
                           spaceAfter=14),
    'h1': ParagraphStyle('h1', fontName='Segoe-Semi', fontSize=15, leading=19, textColor=INK,
                         spaceBefore=14, spaceAfter=6, keepWithNext=1),
    'h2': ParagraphStyle('h2', fontName='Segoe-Semi', fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=10, spaceAfter=4, keepWithNext=1),
    'body': ParagraphStyle('body', fontName='Segoe', fontSize=10, leading=14.5, textColor=INK,
                           spaceAfter=6, alignment=TA_LEFT),
    'bullet': ParagraphStyle('bullet', fontName='Segoe', fontSize=10, leading=14, textColor=INK),
    'caption': ParagraphStyle('caption', fontName='Segoe', fontSize=9, leading=12.5, textColor=INK_2,
                              spaceBefore=4, spaceAfter=10),
    'eq': ParagraphStyle('eq', fontName='Segoe', fontSize=10.5, leading=15, textColor=INK,
                         leftIndent=0.8 * cm, spaceBefore=2, spaceAfter=8),
    'box': ParagraphStyle('box', fontName='Segoe', fontSize=10, leading=14, textColor=INK),
    'cell': ParagraphStyle('cell', fontName='Segoe', fontSize=9, leading=12, textColor=INK),
    'cellh': ParagraphStyle('cellh', fontName='Segoe-Semi', fontSize=9, leading=12, textColor=INK),
}

PAGE_W = A4[0] - 4.0 * cm


def P(text, style='body'):
    return Paragraph(text, S[style])


def bullets(items, style='bullet'):
    return ListFlowable([ListItem(P(t, style), leftIndent=12, value='circle') for t in items],
                        bulletType='bullet', bulletFontName='Segoe', bulletFontSize=8, leftIndent=14,
                        bulletColor=INK_2, spaceAfter=6)


def figure(name, caption, width=PAGE_W):
    path = os.path.join(FIG, name)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(path, width=width, height=width * h / w)
    return KeepTogether([img, P(caption, 'caption')])


def callout(text):
    t = Table([[P(text, 'box')]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PANEL),
        ('LINEBEFORE', (0, 0), (0, -1), 3, ACCENT),
        ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([t, Spacer(1, 8)])


def table(rows, widths):
    data = [[P(c, 'cellh' if i == 0 else 'cell') for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, 0), 0.8, INK_2),
        ('LINEBELOW', (0, 1), (-1, -1), 0.4, RULE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return KeepTogether([t, Spacer(1, 10)])


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Segoe', 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2.0 * cm, 1.2 * cm, 'Semantic Deep JSCC over fading channels: progress update')
    canvas.drawRightString(A4[0] - 2.0 * cm, 1.2 * cm, f'{doc.page}')
    canvas.restoreState()


def build():
    story = []

    # ------------------------------------------------------------------ title
    story += [
        P('Semantic Deep Joint Source-Channel Coding over Fading Channels', 'title'),
        P('Progress update: project pivot, revised goal and first results', 'subtitle'),
        P('October 2026', 'meta'),
    ]

    # ------------------------------------------------------------------ summary
    story += [P('Summary', 'h1')]
    story += [P(
        'This project transmits camera frames over a simulated wireless channel using deep joint '
        'source-channel coding (Deep JSCC): instead of compressing an image into bits and then '
        'protecting those bits with an error-correcting code, a neural network maps the image directly '
        'to channel symbols, and a second network reconstructs the image from the noisy, faded symbols. '
        'The system works in the latent space of a pretrained image autoencoder and is trained to '
        'operate over three channel models with a single set of weights: AWGN, Rician fading '
        '(line-of-sight) and Rayleigh fading (non-line-of-sight).')]
    story += [P(
        'An initial prototype existed but, on review, did not support its central claim: it did not '
        'work on Rayleigh fading. I therefore re-scoped the project around a correct, properly measured '
        'system. This update covers what the first phase did, why the goal changed, the redesigned '
        'pipeline, and two results: a receiver design choice that makes fading channels learnable, '
        'and the trade-off between bandwidth and reconstruction quality.')]
    story += [callout(
        '<b>Key results so far.</b> (1) Feeding raw channel-state information into the transformer '
        'receiver fails on Rayleigh fading: quality stays flat at about 15.8 dB PSNR whatever the SNR. '
        'Equalizing each symbol before the transformer fixes this, and quality then rises steadily with '
        'SNR (17.6 dB at 0 dB to 21.6 dB at 20 dB). (2) Increasing the channel bandwidth from 256 to 1024 '
        'symbols per frame raises quality at every SNR, by about 3 dB at 20 dB. (3) All three channel types '
        'are handled by one model, which encodes, transmits and decodes about 23 frames per second on a '
        'laptop GPU.')]

    # ------------------------------------------------------------------ phase 1
    story += [P('1. Phase 1: the initial prototype', 'h1')]
    story += [P(
        'The original goal was a real-time "6G semantic video" demonstration. Webcam frames were '
        'resized to 128x128 pixels, compressed by the encoder of a frozen Stable Diffusion autoencoder '
        '(VAE) into a 4x16x16 latent, mapped by a vision-transformer (ViT) transmitter to 256 complex '
        'channel symbols, passed through a simulated channel, and decoded by a ViT receiver and the VAE '
        'decoder. The announced novelty was robustness: one trained model working across AWGN, Rician '
        'and Rayleigh channels. The model was trained for 30 epochs on CIFAR-10 images (32x32 pixels, '
        'upsampled to 128x128), and a browser front end streamed live webcam video through it.')]
    story += [P(
        'An audit of the code and the trained model at the end of this phase found the following:', 'body')]
    story += [bullets([
        '<b>The model had not learned Rayleigh fading.</b> On held-out data its reconstruction error on '
        'Rayleigh stayed at the same level for every SNR from 0 to 40 dB, about seven times higher than '
        'on AWGN. Better signal quality did not help at all. Since Rayleigh was half of the robustness '
        'claim, the main claim did not hold.',
        '<b>The SNR axis was off by 3 dB.</b> The power normalization gave each complex symbol a power of 2 '
        'rather than 1, so every reported SNR was 3 dB more optimistic than the true value.',
        '<b>Nothing had been measured in image space.</b> The evaluation and test scripts were broken, there '
        'were no PSNR results, and the live output was a heavily blurred image.',
        '<b>The bandwidth was very aggressive.</b> 256 complex symbols per 128x128x3 frame is about 1/192 '
        'of the pixel values, far tighter than typical Deep JSCC settings, which limits sharpness at any SNR.',
    ])]

    # ------------------------------------------------------------------ pivot
    story += [P('2. Revised goal and why I changed direction', 'h1')]
    story += [P('The revised goal for the current stage of the project is:', 'body')]
    story += [callout(
        'Build a <b>correct and properly evaluated</b> semantic Deep JSCC system that works across AWGN, '
        'Rician and Rayleigh fading with one model; measure it on standard test images; understand which '
        'design choices make it work; and quantify its trade-offs (bandwidth, SNR) against conventional '
        'digital transmission.')]
    story += [P('The reasons for the change:', 'body')]
    story += [bullets([
        '<b>The main claim had to be made true first.</b> A robustness claim cannot be demonstrated with a '
        'model that does not work on one of the three channels. Fixing this exposed a genuine design '
        'insight (Section 4), which is now a central part of the project.',
        '<b>Multi-channel training on its own is not new.</b> Deep JSCC over fading channels with channel '
        'knowledge at the receiver has been studied since Bourtsoulatze, Kurka and Gündüz (2019), and '
        'later work (e.g. attention-based SNR adaptation, transformer-based JSCC) trains over mixed '
        'conditions as standard practice. The value of the project therefore has to come from a sound '
        'design, careful measurement and honest comparison, not from the robustness claim alone.',
        '<b>"Perfectly clear video at any signal quality" is not physically achievable.</b> Shannon '
        'capacity limits how much information a noisy channel can carry. At low SNR and tight bandwidth no '
        'scheme, digital or neural, can deliver a sharp frame. The meaningful question is how gracefully '
        'quality degrades and how it trades against bandwidth, which is what the revised goal measures.',
        '<b>Practical constraints.</b> All training runs on a single laptop GPU (RTX 3060, 6 GB). Caching '
        'the VAE latents once and training only the transmitter and receiver makes each training run take '
        'roughly one to three hours, which allows controlled experiments such as ablations and bandwidth '
        'sweeps.',
    ])]
    story += [P('Changes made to the system as part of the pivot:', 'body')]
    story += [table([
        ['Area', 'Phase 1', 'Now'],
        ['Receiver input', 'Raw received symbols and channel coefficients, concatenated',
         'Per-symbol MMSE equalization first, then the transformer (Section 4)'],
        ['SNR definition', 'Symbol power 2, SNR labels 3 dB optimistic', 'Unit symbol power; SNR is exact'],
        ['SNR during training', 'One random SNR per batch', 'Independent random SNR per sample'],
        ['Bandwidth', 'Fixed at 256 symbols per frame', 'Configurable; 256 / 512 / 1024 compared (Section 5)'],
        ['Evaluation', 'None in image space', 'PSNR / SSIM on the Kodak test set, per channel and SNR'],
        ['Engineering', 'Broken tests, no version control',
         'Test suite, versioned checkpoints with stored configuration, resumable training, Git'],
    ], [3.3 * cm, 5.9 * cm, 7.8 * cm])]

    # ------------------------------------------------------------------ system
    story += [PageBreak(), P('3. System overview', 'h1')]
    story += [figure('fig1_system.png', width=PAGE_W * 0.82, caption=
                     '<b>Figure 1.</b> Pipeline. The signal flows left to right along the top row, through '
                     'the channel, then right to left along the bottom row. Grey blocks are the frozen, '
                     'pretrained Stable Diffusion VAE; blue blocks are trained end to end over randomly drawn '
                     'channel types and SNRs.')]
    story += [P(
        '<b>Source side.</b> A 128x128 RGB frame is encoded by the frozen VAE into a 4x16x16 latent. Working '
        'in this latent space means the trainable part only has to transmit a compact semantic '
        'representation, and the VAE decoder turns the received latent back into a natural-looking image.')]
    story += [P(
        '<b>Transmitter.</b> A 4-layer vision transformer splits the latent into 2x2 patches (64 tokens), '
        'processes them with self-attention, and projects the result to K complex channel symbols. A power '
        'normalization layer scales every frame so the average symbol power is exactly 1, so the network '
        'cannot "beat" the noise by transmitting louder.')]
    story += [P('<b>Channel.</b> Each symbol x passes through', 'body')]
    story += [P('y = h · x + n,  &nbsp;&nbsp; n ~ CN(0, σ<super>2</super>),  &nbsp;&nbsp; '
                'SNR = 1 / σ<super>2</super>', 'eq')]
    story += [P(
        'with h = 1 for AWGN, h ~ CN(0, 1) for Rayleigh fading, and a Rician coefficient with K-factor 3 '
        '(a strong line-of-sight component plus scattering). In both fading cases E|h|<super>2</super> = 1, '
        'and h is drawn independently for every symbol (fast fading). The receiver is assumed to know h '
        '(perfect channel-state information) and the SNR.')]
    story += [P(
        '<b>Receiver.</b> Each received symbol is first equalized with the minimum mean-square-error (MMSE) '
        'estimator', 'body')]
    story += [P('x<sub>eq</sub> = y · h* / ( |h|<super>2</super> + σ<super>2</super> )', 'eq')]
    story += [P(
        'and a mirror-image 4-layer transformer maps the features [Re x<sub>eq</sub>, Im x<sub>eq</sub>, '
        '|h|<super>2</super>, SNR] for all symbols back to a 4x16x16 latent, which the VAE decodes into the '
        'image. Including |h|<super>2</super> and the SNR tells the network how reliable each equalized '
        'symbol is.')]
    story += [P(
        '<b>Training.</b> The VAE latents of the training images are computed once and cached. Only the '
        'transmitter and receiver are trained, with AdamW (learning rate 3x10<super>-4</super>, cosine '
        'decay), batch size 64, for 30 epochs. Each batch uses a random channel type; each sample uses a '
        'random SNR between 0 and 20 dB. The loss is the mean-squared error between the original and '
        'reconstructed latents plus a small cosine-similarity term. <b>Evaluation</b> uses the 24 Kodak '
        'images, a standard image-transmission benchmark, cut into 144 tiles of 128x128 pixels, with '
        'reconstruction quality reported as PSNR in the image domain.')]

    # ------------------------------------------------------------------ equalization
    story += [PageBreak(), P('4. Result 1: explicit equalization makes Rayleigh fading learnable', 'h1')]
    story += [figure('fig2_equalization.png',
                     '<b>Figure 2.</b> PSNR versus channel SNR on Rayleigh fading (K = 256 symbols, Kodak, '
                     'models trained on CIFAR-10 latents). Both models have the same architecture, training '
                     'data and schedule; only the receiver input differs.', width=PAGE_W * 0.92)]
    story += [P(
        '<b>What the figure shows.</b> With the original design, where the received symbols y and the '
        'channel coefficients h are simply concatenated and fed to the transformer, quality on Rayleigh '
        'fading stays almost flat: 15.2 dB PSNR at 0 dB SNR and only 15.8 dB at 20 dB. A 100-fold '
        'improvement in signal quality buys about half a decibel. With per-symbol MMSE equalization in '
        'front of the same transformer, quality rises steadily with SNR, from 17.6 dB to 21.6 dB, and is '
        'higher at every SNR.')]
    story += [P(
        '<b>Why it happens.</b> Rayleigh fading multiplies every symbol by a random complex number, which '
        'rotates it by a random phase and scales it by a random gain. Undoing this requires dividing y by h '
        'for each symbol: a multiplicative operation. The receiver\'s first layer is linear and mixes all '
        'symbols together into tokens, so it cannot represent a per-symbol division, and the attention '
        'layers that follow see only scrambled combinations. The network therefore cannot recover the '
        'phase, and the extra signal quality is wasted. AWGN and Rician fading hide this problem: with h = 1, '
        'or with h dominated by a fixed line-of-sight component, there is little phase to undo.')]
    story += [P(
        '<b>Why it matters.</b> This was the reason the Phase 1 model failed on Rayleigh. The fix costs a few '
        'lines of code and no extra parameters: classical signal processing handles the part it is optimal '
        'for (undoing a known channel), and the network learns the rest (decoding the semantic content and '
        'weighting unreliable symbols using |h|<super>2</super> and the SNR). The controlled ablation, '
        'identical except for the receiver input, isolates the effect.')]

    # ------------------------------------------------------------------ bandwidth
    story += [PageBreak(), P('5. Result 2: more channel symbols per frame give sharper reconstructions', 'h1')]
    story += [figure('fig3_bandwidth.png',
                     '<b>Figure 3.</b> PSNR versus channel SNR on Rayleigh fading for three bandwidths '
                     '(Kodak, models trained on CIFAR-10 latents). The dotted line is the quality of the VAE '
                     'alone with no channel: the ceiling for any scheme built on this autoencoder.',
                     width=PAGE_W * 0.92)]
    story += [P(
        '<b>What the figure shows.</b> The bandwidth K is the number of complex channel symbols used per '
        'frame. Each model was trained identically except for K.', 'body')]
    story += [table([
        ['Symbols per frame (K)', 'Fraction of pixel values', 'PSNR at 0 dB', 'PSNR at 10 dB', 'PSNR at 20 dB'],
        ['256', '1/192', '17.6 dB', '20.2 dB', '21.6 dB'],
        ['512', '1/96', '17.8 dB', '20.8 dB', '23.2 dB'],
        ['1024', '1/48', '18.6 dB', '22.1 dB', '24.6 dB'],
        ['VAE alone (no channel)', '-', '26.9 dB', '26.9 dB', '26.9 dB'],
    ], [4.2 * cm, 3.6 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm])]
    story += [P(
        '<b>Interpretation.</b> More bandwidth helps at every SNR, and the benefit grows with SNR: at 20 dB, '
        'going from 256 to 1024 symbols adds 3 dB, while at 0 dB it adds 1 dB. This matches information '
        'theory. At low SNR each symbol carries little information, so noise is the bottleneck whatever '
        'the bandwidth. At high SNR each symbol can carry much more, so the number of symbols becomes the '
        'bottleneck. The K = 256 curve flattens near 21.5 dB because the model runs out of bandwidth, not '
        'signal quality, which is why the Phase 1 demonstration looked blurry even at high SNR.')]
    story += [P(
        'The curves are still below the VAE ceiling of 26.9 dB, so the autoencoder is not yet the limiting '
        'factor at K = 1024; the transmission itself is. On a real webcam frame from the live demo, over '
        'Rayleigh fading at 20 dB SNR, the K = 1024 model reaches 30.0 dB (VAE ceiling for that frame: '
        '32.0 dB), compared with 24.7 dB for K = 256, and faces and objects are clearly recognizable. The '
        'pipeline encodes, transmits and decodes about 23 frames per second on the laptop GPU.')]

    # ------------------------------------------------------------------ next steps
    story += [PageBreak(), P('6. Limitations and next steps', 'h1')]
    story += [KeepTogether([P('<b>Current limitations</b>, to be stated in the final report:', 'body'), bullets([
        'The channel is simulated, with perfect channel knowledge at the receiver and independent fading '
        'per symbol. Real links have imperfect channel estimates and fading that is correlated in time.',
        'Frames are 128x128 pixels, which limits fine detail regardless of channel quality.',
        'PSNR measures pixel accuracy, not how natural an image looks; a perceptual metric is needed '
        'alongside it.',
    ])])]
    story += [P('<b>Next steps for the final submission:</b>', 'body')]
    story += [bullets([
        '<b>Comparison with conventional digital transmission.</b> Benchmark the system against separate '
        'source and channel coding (image compression plus a channel code) at the same bandwidth, both '
        'with an idealized capacity-achieving code and with a practical set of modulation and coding '
        'rates. The key question is where Deep JSCC degrades gracefully and where a digital link performs '
        'better. These experiments are set up and the analysis is in progress.',
        '<b>Training on natural photographs.</b> Retrain on crops of the DIV2K photo dataset instead of '
        'upsampled CIFAR-10 images, to better match real camera frames, and evaluate the gain.',
        '<b>Visual and perceptual evaluation.</b> Side-by-side reconstructions across SNRs and channels, '
        'plus a perceptual metric (LPIPS) alongside PSNR and SSIM.',
        '<b>Live demonstration.</b> The browser demo streams webcam video through the system with selectable '
        'channel type and SNR, alongside a digital baseline at the same bandwidth.',
        '<b>Final report</b> with all figures, the ablation, and the limitations above.',
    ])]
    story += [P('<b>Longer-term direction</b> (beyond this submission):', 'body')]
    story += [P(
        'At low SNR, a decoder trained to minimize pixel error must output a blurred average, so images '
        'look foggy. A natural follow-up is a <b>real-time generative video JSCC</b>: a decoder trained for '
        'perceptual quality, conditioned on previous frames so that it only needs to transmit what changed. '
        'Recent generative JSCC methods are mostly slow, multi-step and image-only, while existing video '
        'JSCC methods optimize pixel error. A system that is real-time, temporally conditioned and '
        'perceptually sharp at low SNR is a plausible research contribution, subject to a full literature '
        'review and access to more GPU compute.')]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                            topMargin=1.8 * cm, bottomMargin=2.0 * cm,
                            title='Semantic Deep JSCC over Fading Channels: Progress Update',
                            author='', subject='Project progress update')
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    build()
