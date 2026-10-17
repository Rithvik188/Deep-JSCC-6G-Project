"""
server.py
---------
Live demo: webcam frames -> Semantic Deep JSCC over a simulated AWGN/Rician/Rayleigh channel,
shown next to a digital baseline (WebP over a practical channel code) using the SAME number of
channel symbols, so the digital "cliff effect" is visible live.

Usage:
  python server.py --checkpoint checkpoints/k1024_best.pth [--port 8001] [--no-fp16]
"""

import os
import io
import math
import json
import time
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torchvision.transforms as transforms

from models.jscc_system import build_from_checkpoint
from baselines.digital import digital_transmit

app = FastAPI(title="Semantic Deep JSCC Backend")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set from the command line in __main__ before the app starts
settings = {'checkpoint': 'checkpoints/best_jscc_model.pth', 'fp16': True}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
digital_pool = ThreadPoolExecutor(max_workers=1)  # digital baseline runs on CPU alongside the GPU work

CHANNELS = ['awgn', 'rician', 'rayleigh']
DIGITAL_MODES = {'adaptive': None, 'fixed10': 10.0}  # design SNR for the practical digital link

# Image transforms
img_size = 128
to_tensor = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()


@app.on_event("startup")
async def startup_event():
    global model
    print(f"Loading Semantic JSCC Model on {device}...")
    # Fails loudly rather than silently streaming output from an untrained model
    model, epoch = build_from_checkpoint(settings['checkpoint'], device, use_vae=True)
    if settings['fp16'] and device.type == 'cuda':
        model.vae.half()
    print(f"Loaded {settings['checkpoint']}: {model.config} (epoch {epoch + 1}), "
          f"VAE in {'fp16' if settings['fp16'] and device.type == 'cuda' else 'fp32'}.")


@app.get("/config")
async def get_config():
    return {"num_symbols": model.config['num_symbols'], "checkpoint": os.path.basename(settings['checkpoint'])}


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to video stream.")

    try:
        while True:
            # Receive JSON payload from client
            data_str = await websocket.receive_text()
            try:
                reply = process_frame(json.loads(data_str))
            except Exception as e:
                # A bad frame should not kill the stream; reply so the client keeps looping
                print(f"Frame error: {e}")
                await websocket.send_json({"error": str(e)})
                continue
            await websocket.send_json(reply)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")


def to_data_uri(tensor):
    buffer = io.BytesIO()
    to_pil(tensor.clamp(0, 1).float().cpu()).save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


@torch.no_grad()
def run_jscc(x, snr_db, channel_type):
    vae = model.vae
    dtype = next(vae.parameters()).dtype
    latents = vae.encode(x.to(dtype)).float()
    latents_hat = model.forward_latent(latents, snr_db=snr_db, channel_type=channel_type)
    return vae.decode(latents_hat.to(dtype)).float()


def process_frame(data):
    """Decodes one client frame, runs JSCC (GPU) and the digital baseline (CPU) in parallel."""
    t0 = time.time()
    base64_image = data.get("image", "")
    snr_db = float(data.get("snr", 10.0))
    if not math.isfinite(snr_db):
        snr_db = 10.0
    # Stay inside the SNR range the model was trained on
    snr_db = min(max(snr_db, 0.0), 20.0)
    channel_type = data.get("channel", "rician")
    if channel_type not in CHANNELS:
        channel_type = 'rician'
    digital_mode = data.get("digital", "fixed10")
    if digital_mode not in DIGITAL_MODES:
        digital_mode = 'fixed10'

    # Decode base64 image
    if base64_image.startswith("data:image"):
        base64_image = base64_image.split(",")[1]
    pil_image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
    x = to_tensor(pil_image).unsqueeze(0)

    digital_future = digital_pool.submit(
        digital_transmit, x[0], model.config['num_symbols'], snr_db, channel_type,
        design_snr_db=DIGITAL_MODES[digital_mode], mode='practical')

    x_hat = run_jscc(x.to(device), snr_db, channel_type)
    digital_img, digital_bytes = digital_future.result()

    return {
        "image_reconstructed": to_data_uri(x_hat[0]),
        "image_digital": to_data_uri(digital_img),
        "digital_bytes": digital_bytes,
        "num_symbols": model.config['num_symbols'],
        "snr": snr_db,
        "server_ms": round((time.time() - t0) * 1000),
    }


# Mount the frontend directory to serve the UI directly
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=settings['checkpoint'])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-fp16", action="store_true", help="Run the VAE in fp32")
    args = parser.parse_args()
    settings['checkpoint'] = args.checkpoint
    settings['fp16'] = not args.no_fp16

    ssl_args = {}
    if os.path.exists("key.pem") and os.path.exists("cert.pem"):
        ssl_args = {"ssl_keyfile": "key.pem", "ssl_certfile": "cert.pem"}
    print(f"Starting {'HTTPS' if ssl_args else 'HTTP'} server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, **ssl_args)
