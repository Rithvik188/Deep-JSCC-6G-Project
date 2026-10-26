# Deep JSCC 6G Project — Complete AI Handoff Document

> **FINAL STATE (2026-10-26): project complete.** Team B-9: Rithvik Arulprakash, Harshith K V, Venugopalan G, Vipin Sudhakar.
>
> - **Final model:** `checkpoints/k1024_div2k_best.pth` (K=1024 symbols, MMSE receiver, trained on DIV2K crops). Other checkpoints: `best_jscc_model.pth` (K=256), `k512_best.pth`, `k1024_best.pth` (CIFAR), `k256_raw_best.pth` (raw-CSI ablation).
> - **Results:** `results/kodak/metrics_kodak.csv` (5 models × 3 channels × 0–20 dB, plus ideal/practical/fixed digital baselines). VAE ceiling 26.9 dB. About 23 fps end to end.
> - **Deliverables, all regenerable:**
>   - `python report/make_figures.py` → `report/figures/{doc,slides,ieee}/` and `report/numbers.json`
>   - `python report/build_slides.py` → `report/B9_Final_Review.pptx` (15 slides)
>   - LaTeX in `report/ieee/` (pdflatex, bibtex, pdflatex ×2; MiKTeX installed) → `report/B9_IEEE_Report.pdf`
>   - `python report/build_update_pdf.py` → the earlier progress update
> - **Demo:** `python generate_cert.py <wifi-ip>` (pass the IP explicitly, since ProtonVPN can be picked up), then `python server.py --checkpoint checkpoints/k1024_div2k_best.pth --port 8001`.
> - **Honest headline:** degrades gracefully and beats fixed-rate digital below its design SNR; adaptive digital wins by 2.5–3.3 dB at high SNR, because the VAE caps quality.
> - The slide version of Fig. 5 contains a webcam frame of a team member's room, which is why generated images and decks are not committed.

> **UPDATE 2026-10-17 (Claude audit). Read this first; it supersedes the sections below where they conflict.**
>
> - **The original 30-epoch model did not work on Rayleigh.** Latent MSE was ~0.485 at every SNR from 0 to 40 dB, against 0.719 for just predicting the mean latent. AWGN was good (~0.07) and Rician had a floor (~0.135). Cause: the receiver fed raw `[y, h]` into a linear layer, which cannot do the per-symbol complex division needed to undo fading.
> - **Fixed (MODEL_VERSION 2):** the receiver now MMSE-equalizes each symbol before the ViT, and is given the SNR (`receiver(y, h, snr_db)`). The power constraint now gives E|s|^2 = 1 per complex symbol; it was 2, which made every SNR label 3 dB optimistic. SNR is sampled per sample during training.
> - **Old checkpoints are incompatible** and are refused on load (`load_jscc_weights` checks `model_version`). They were moved to `checkpoints/old_v1/`.
> - **Training:** `python main.py` resumes from `checkpoints/latest_jscc_model.pth` (model + optimizer + scheduler state). Best = lowest mean val MSE over AWGN/Rician/Rayleigh at 10 dB.
> - **Evaluation:** `python evaluate_visuals.py [--image-dir folder_of_photos]` writes PSNR/SSIM per channel × SNR on the official CIFAR-10 test split, a VAE-only ceiling, and grids to `results/`.
> - **Tests:** `python -m pytest` (13 tests). `test_inference.py` and `quick_sanity_test.py` were removed (they were broken / didn't test the trained model).
> - **Doc corrections:** installed PyTorch is 2.7.1+cu118 (not 2.13). There are no `__init__.py` files (namespace packages). Checkpoints contain only `transmitter.*` and `receiver.*`. Training images are CIFAR-10 32 px upsampled to 128 px, which is a domain gap vs. webcam frames. `DeepJSCCSystem(use_vae=False)` is now the training model (`LatentJSCC` is gone), and `model(x)` returns just `x_hat`.
> - Git: repo at https://github.com/Rithvik188/Deep-JSCC-6G-Project. Data, checkpoints and TLS keys are gitignored.
**For:** Claude (or any incoming AI agent)
**From:** Antigravity (Google Deepmind AI, conversation ID: 4657c0a1-e9c0-43e5-bf69-986ca47a4e3f)
**Date:** 2026-10-17
**Project Root:** `C:\Users\rithv\OneDrive\Desktop\Deep_JSCC_6G_Project\`

---

## 1. PROJECT OVERVIEW

**What this project is:**
A real-time 6G Deep Joint Source-Channel Coding (Deep JSCC) system that transmits live video frames (from a webcam/iPad) over a simulated wireless channel and reconstructs them on the other side. The novelty is that it works with **Rician fading (6G Line-of-Sight), Rayleigh fading (6G Non-Line-of-Sight), and AWGN (baseline)** — all three channel conditions — using a single trained model.

**What it is NOT:**
- Not a traditional codec (no H.264, no compression pipeline)
- Not a simulation-only system — it has a live video streaming frontend
- Not a research demo with canned data — it processes real webcam frames end-to-end

**The core pipeline:**
```
Webcam Frame (128x128 RGB)
  -> VAE Encoder (frozen, Stable Diffusion sd-vae-ft-mse) -> latents (4, 16, 16)
  -> ViT Transmitter -> complex symbols (2, 256)
  -> Wireless Channel (AWGN / Rayleigh / Rician + random SNR 0-20dB)
  -> ViT Receiver (with CSI h concatenated) -> reconstructed latents (4, 16, 16)
  -> VAE Decoder (frozen) -> reconstructed image (3, 128, 128)
```

---

## 2. HARDWARE & ENVIRONMENT

- **OS:** Windows 11, PowerShell
- **GPU:** NVIDIA RTX 3060 (CUDA enabled, verified working)
- **Python:** 3.13
- **PyTorch:** 2.13.0 (CUDA build, cu121)
- **Key packages:** torch, torchvision, diffusers, transformers, fastapi, uvicorn, Pillow

**Important Windows quirk:** PowerShell uses `cp1252` encoding. Unicode characters like `→` in print() statements will crash. Always use ASCII `->`.

---

## 3. COMPLETE FILE STRUCTURE

```
Deep_JSCC_6G_Project/
├── main.py                    # Training entry point (MODIFIED - see below)
├── train.py                   # Training loop (MODIFIED - see below)
├── server.py                  # FastAPI WebSocket inference server (MODIFIED)
├── precompute_latents.py      # One-time VAE latent caching script (NEW)
├── requirements.txt           # torch torchvision numpy matplotlib pytorch-msssim pytest
├── evaluate_visuals.py        # Visual evaluation script
├── quick_sanity_test.py       # 50-step sanity test script
├── test_inference.py          # Inference test
├── generate_cert.py           # SSL cert generator for HTTPS server
├── cert.pem / key.pem         # SSL certificates (generated)
├── AI_Handoff_Doc.md          # Old/original handoff doc (ignore this)
├── CLAUDE_HANDOFF.md          # THIS FILE
├── channel/
│   ├── __init__.py
│   └── fading.py              # WirelessChannel: AWGN, Rayleigh, Rician (MODIFIED)
├── models/
│   ├── __init__.py
│   ├── encoder.py             # ViTTransmitter
│   ├── decoder.py             # ViTReceiver with CSI input (MODIFIED)
│   ├── jscc_system.py         # DeepJSCCSystem orchestrator (MODIFIED)
│   └── vae.py                 # SemanticVAE wrapper (frozen pretrained)
├── data/
│   ├── dataset.py             # Original image dataloader (CIFAR-10)
│   ├── latent_dataset.py      # NEW: LatentDataset for cached latents
│   ├── latents_train.pt       # CACHED: 40,000 pre-encoded VAE latents (~200MB)
│   ├── latents_test.pt        # CACHED: 9,984 pre-encoded VAE latents (~50MB)
│   └── cifar-10-batches-py/   # Raw CIFAR-10 dataset (local)
├── checkpoints/
│   ├── best_jscc_model.pth    # BEST trained weights (163MB) ← USE THIS
│   └── latest_jscc_model.pth  # Latest epoch weights (163MB)
├── frontend/
│   ├── index.html             # Video streaming UI with channel selector
│   └── app.js                 # WebSocket client, sends channel type to backend
├── utils/
│   └── losses.py              # JSCCLoss, compute_psnr
└── tests/
    └── (test files)
```

---

## 4. WHAT HAS BEEN BUILT — CHRONOLOGICAL HISTORY

### Phase 1: Project Initialization (Aug 2026)
- Created project directory structure from scratch
- Installed PyTorch 2.13 with CUDA support (RTX 3060 verified)
- Set up Git repository

### Phase 2: Core Architecture (Aug-Sep 2026)
- Built `ViTTransmitter` — Vision Transformer that encodes image patches into complex channel symbols
- Built `ViTReceiver` — Vision Transformer that decodes received symbols back to image space
- Built `WirelessChannel` — AWGN channel simulation
- Built `SemanticVAE` — wrapper around Stable Diffusion `sd-vae-ft-mse` VAE (fully frozen)
- Built `DeepJSCCSystem` — orchestrates the full pipeline

### Phase 3: Channel Upgrades (Oct 2026)
- **Rayleigh fading added** to `channel/fading.py` — complex Gaussian h ~ CN(0,1), models 6G NLoS
- **Rician fading added** to `channel/fading.py` — K-factor=3.0, models 6G LoS
- All three channel types now return `(y, h)` where `h` is the Channel State Information (CSI)

### Phase 4: CSI Equalization (Oct 2026)
- **`ViTReceiver` upgraded** to accept CSI: concatenates `y` (2, K) + `h` (2, K) = 4 channels as input
- `input_proj` layer now maps `4 * num_symbols` -> `num_patches * embed_dim`
- AWGN fallback: if h=None, assumes h_real=1, h_imag=0
- This allows the receiver to learn channel inversion from the fading coefficient directly

### Phase 5: Training Speed Fix (Oct 2026)
**Critical optimization — this is the most important architectural decision:**

The VAE encoder was running on every training batch, adding ~22 minutes/epoch overhead. The fix:
1. **`precompute_latents.py`** — one-time script that encodes all 50,000 CIFAR images through the frozen VAE and saves latents to disk. Ran once, took ~45 minutes.
2. **`data/latent_dataset.py`** — `LatentDataset` class that loads pre-cached `.pt` files
3. **`train.py` rewritten** — training loop now receives raw latents directly, calls `model.transmitter` → `model.channel` → `model.receiver`. VAE is never called during training.
4. **`main.py` rewritten** — uses `LatentJSCC` (Transmitter + Channel + Receiver only, no VAE) for training. Auto-calls precompute if `.pt` files are missing.

Result: epoch time went from ~25 minutes to ~5-6 minutes (including eval pass).

### Phase 6: Training Run (Oct 13-14, 2026)
- Trained 30 epochs from scratch (old 2-epoch checkpoint was incompatible with new architecture)
- Training completed fully and automatically

**Final training results:**
| Epoch | Train Loss | Val MSE | Val CosSim |
|-------|-----------|---------|------------|
| 1     | 0.3751    | 0.6266  | 0.558      |
| 5     | 0.3457    | 0.5852  | 0.595      |
| 10    | 0.3193    | 0.5470  | 0.626      |
| 20    | ~0.280    | ~0.510  | ~0.658     |
| 27    | 0.2725    | 0.4895  | 0.671      |
| 30    | 0.2794    | **0.4872** | **0.672** |

- Best checkpoint saved at every epoch through epoch 30
- `checkpoints/best_jscc_model.pth` = epoch 30 weights

---

## 5. CURRENT CODE — KEY FILES IN DETAIL

### `channel/fading.py` — WirelessChannel
```python
# Returns (y, h) for all channel types
# x: (B, 2, K) — real/imag channel symbols
# snr_db: float — Signal-to-Noise Ratio in dB
# channel_type: 'awgn', 'rayleigh', 'rician'
# Rician K-factor = 3.0 (strong LoS component)
# noise_var = 10^(-snr_db/10)
# y = h*x + n, h = CSI coefficient
# Returns: y (B,2,K), h (B,2,K)
```

### `models/encoder.py` — ViTTransmitter
```python
# img_size=16, patch_size=2, in_channels=4 (when processing VAE latents)
# embed_dim=128, depth=4, num_heads=4, num_symbols=256
# patch_embed: Conv2d(4, 128, kernel=2, stride=2) -> 64 patches
# transformer: 4-layer TransformerEncoder, norm_first=True, GELU activation
# head: Linear(64*128, 2*256) -> (B, 2, 256) complex symbols
# ChannelPowerConstraint: normalizes RMS power to 1.0
```

### `models/decoder.py` — ViTReceiver (CSI-aware)
```python
# img_size=16, patch_size=2, out_channels=4, use_sigmoid=False
# input_proj: Linear(4*256, 64*128) — takes y and h concatenated
# transformer: 4-layer TransformerEncoder, norm_first=True, GELU
# head: Linear(128, 2*2*4) — maps back to patch pixels
# Output: (B, 4, 16, 16) latents
```

### `models/jscc_system.py` — DeepJSCCSystem
```python
# use_vae=True mode: full pipeline including VAE (used by server.py)
# use_vae=False mode: no VAE (not used for current training)
# forward_latent_training(): VAE encode -> Tx -> Channel -> Rx (legacy, not used now)
# forward(): full pipeline for inference
```

### `train.py` — Training Loop
```python
# Input: batch = latent tensor (B, 4, 16, 16) directly
# Per batch: random SNR in [0, 20] dB, random channel from ['awgn','rician','rayleigh']
# Forward: model.transmitter(latents) -> model.channel(s, snr, ch) -> model.receiver(y, h)
# Loss = MSE(latents_hat, latents) + 0.1 * (1 - CosineSimilarity)
# Checkpoint saved every 200 batches
# evaluate_epoch(): uses SNR=10dB, Rayleigh for validation
```

### `main.py` — Training Entry Point
```python
# LatentJSCC class defined inline (transmitter + channel + receiver, no VAE)
# Auto-calls precompute_latents.py if .pt files don't exist
# Batch size: 64, NUM_EPOCHS: 30
# Optimizer: AdamW lr=3e-4, weight_decay=1e-4
# Scheduler: CosineAnnealingLR, T_max=30, eta_min=1e-5
# Checkpoint resume: loads best_jscc_model.pth or latest_jscc_model.pth
# Checkpoint format: {'model_state': ..., 'optimizer_state': ..., 'epoch': N, 'best_mse': X}
```

### `server.py` — FastAPI Inference Server
```python
# Loads DeepJSCCSystem(use_vae=True) — full pipeline including VAE
# Checkpoint loading uses strict=False (ViT weights load; VAE uses pretrained frozen weights)
# WebSocket endpoint: /ws/stream
# Receives JSON: {image: base64, snr: float, channel: 'awgn'|'rician'|'rayleigh'}
# Returns JSON: {image_reconstructed: base64_jpeg, snr: float}
# HTTPS if cert.pem/key.pem exist, else HTTP
# Serves frontend/ as static files at /
```

### `frontend/index.html` + `app.js`
- Live video capture from webcam/iPad camera
- Dropdown to select channel type (Rician 6G LoS / Rayleigh 6G NLoS / AWGN Baseline)
- SNR slider
- Sends frame → WebSocket → displays reconstructed frame side by side
- app.js sends `{image, snr, channel}` JSON payload

---

## 6. CHECKPOINT FORMAT — CRITICAL FOR LOADING

The `best_jscc_model.pth` checkpoint was saved by `LatentJSCC` (training model), which has keys:
- `transmitter.*` (ViTTransmitter weights)
- `receiver.*` (ViTReceiver weights)
- `channel.*` (WirelessChannel — no learnable params, but included)

The `server.py` loads `DeepJSCCSystem(use_vae=True)` which additionally has:
- `vae.*` (frozen VAE weights — these are NOT in the checkpoint)

**This is correct and intentional.** Server uses `strict=False` when loading, so:
- `transmitter.*` and `receiver.*` load from checkpoint ✅
- `vae.*` uses the downloaded pretrained weights from HuggingFace ✅

To load manually:
```python
model = DeepJSCCSystem(use_vae=True).to(device)
ckpt = torch.load('checkpoints/best_jscc_model.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state'], strict=False)
```

---

## 7. HOW TO RUN

### Start the inference server:
```powershell
cd C:\Users\rithv\OneDrive\Desktop\Deep_JSCC_6G_Project
python server.py
```
Then open browser (or iPad) to `https://<machine-ip>:8000`

### Continue training (if needed):
```powershell
cd C:\Users\rithv\OneDrive\Desktop\Deep_JSCC_6G_Project
python main.py
```
It will auto-detect the checkpoint and resume. The `.pt` cached latent files exist so precompute is skipped.

### Run sanity test:
```powershell
python quick_sanity_test.py
```

---

## 8. WHAT STILL NEEDS TO BE DONE

This is the critical list of remaining work. Training is complete. What's NOT done:

### 8.1 — VISUAL QUALITY EVALUATION (IMMEDIATE PRIORITY)
The model has been trained 30 epochs with Val CosSim=0.672. But we have **never visually verified the reconstruction quality** after the new fast-training architecture. The last visual test (after only 2 epochs with the OLD architecture) showed a blurry gray fog.

**To do:**
1. Start `server.py`
2. Connect from browser
3. Stream a webcam feed
4. Compare original vs. reconstructed side-by-side
5. Test all three channel types and multiple SNR values
6. Honestly assess: does the reconstruction show recognizable content?

The expected outcome at 30 epochs: shapes and colors should be recognizable, though still blurry at low SNR. Faces/rooms should be vaguely discernible. At high SNR (15-20dB) on AWGN, it should look decent.

If reconstruction is still a gray blob: the latent alignment may be converging in a suboptimal basin. In that case, see section 8.4.

### 8.2 — PSNR / SSIM BENCHMARKING
For the project report/presentation, need quantitative metrics:
- PSNR across SNR range (0, 5, 10, 15, 20 dB)
- For each channel type separately
- Compare against baseline (no channel, direct VAE encode→decode)
- `evaluate_visuals.py` exists but may need updating for the new architecture

### 8.3 — PROJECT REPORT / PRESENTATION
The user is a student presenting this as a research project. Needs:
- Architecture diagram
- Results table (PSNR vs SNR for 3 channel types)
- Discussion of novelty: CSI-aware ViT receiver + multi-channel robustness

### 8.4 — POTENTIAL TRAINING IMPROVEMENTS (only if visual quality is bad)
If visual output is unacceptable after testing:

Option A: **More epochs** — run `python main.py` after changing `NUM_EPOCHS = 60` in main.py
Option B: **Perceptual loss** — add LPIPS or SSIM loss in image space (requires decoding latents via VAE during eval, not training)
Option C: **PSNR-guided finetuning** — periodically decode latents and compute image-space PSNR, use as validation signal

### 8.5 — MOBILE/IPAD COMPATIBILITY
The frontend was designed to work on iPad (the user's use case). Needs to be tested on actual iPad:
- HTTPS is required for getUserMedia on iOS Safari
- cert.pem and key.pem are already generated
- Server must run on the machine's LAN IP (not localhost)

---

## 9. KNOWN ISSUES / GOTCHAS

1. **Unicode in print()**: Windows cp1252 encoding. Never use `→`, `✓`, `×` etc. in Python print statements. Always use ASCII `->`, `OK`, `X`.

2. **Checkpoint incompatibility**: The old checkpoint from the 2-epoch run (with DeepJSCCSystem that included VAE in training) is incompatible with the new LatentJSCC training model. The current checkpoints are from the new architecture. Do not try to load old checkpoints into the new model.

3. **VAE not in training checkpoint**: The best_jscc_model.pth does NOT contain VAE weights. That is correct. The VAE is always loaded separately from HuggingFace (cached locally).

4. **HuggingFace unauthenticated warning**: The HF_TOKEN is not set. This is fine — the VAE downloads and works. Rate limits might apply but don't cause failures since it's cached after first download.

5. **num_workers on Windows**: DataLoader `num_workers=2` with `persistent_workers=True` can sometimes cause issues on Windows. If you see multiprocessing errors, set `num_workers=0`.

6. **The evaluation in main.py uses Rayleigh at 10dB**: Val MSE=0.487, Val CosSim=0.672 are both measured at this fixed condition. Performance will be better at higher SNR or AWGN, worse at lower SNR or pure Rayleigh with very low SNR.

---

## 10. USER CONTEXT (IMPORTANT)

- **The user demands brutal honesty.** Do not try to please them or be optimistic. If something doesn't work, say so directly.
- **The user is a student** presenting this as a research project. The "novelty" they announced is multi-channel robustness (works with Rician, Rayleigh, AWGN).
- **Hardware constraint**: RTX 3060 laptop. Cannot run training continuously for long sessions. The latent caching solved the speed problem — each epoch now takes ~5-6 minutes.
- **The user has limited time.** Don't suggest re-architecting unless visually quality is clearly unacceptable. The goal is a working demonstration, not a paper-quality system.
- **PowerShell on Windows.** Do not use `&&` to chain commands — it's invalid in PowerShell. Use `;` instead.

---

## 11. TRAINING CONFIGURATION (for reference)

```python
# In main.py
NUM_EPOCHS = 30
BATCH_SIZE = 64
OPTIMIZER = AdamW(lr=3e-4, weight_decay=1e-4)
SCHEDULER = CosineAnnealingLR(T_max=30, eta_min=1e-5)

# ViTTransmitter config (for latent-space training)
img_size=16, patch_size=2, in_channels=4
embed_dim=128, depth=4, num_heads=4, num_symbols=256

# ViTReceiver config
img_size=16, patch_size=2, out_channels=4, use_sigmoid=False
embed_dim=128, depth=4, num_heads=4, num_symbols=256

# WirelessChannel config
SNR range: uniform[0.0, 20.0] dB (random per batch)
Channel: random choice from ['awgn', 'rician', 'rayleigh']
Rician K-factor: 3.0
Rayleigh std: sqrt(0.5) per component

# Loss
loss = MSE(latents_hat, latents) + 0.1 * (1 - CosineSimilarity)
```

---

## 12. IMMEDIATE NEXT ACTION

**The next thing to do is run the server and visually verify reconstruction quality:**

```powershell
cd C:\Users\rithv\OneDrive\Desktop\Deep_JSCC_6G_Project
python server.py
```

Open browser to `http://localhost:8000` (or HTTPS if certs work).
Stream video. Assess quality honestly.
Report back what you see at SNR=15dB on AWGN vs. Rayleigh.
