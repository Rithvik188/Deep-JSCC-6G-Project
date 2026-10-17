import os
import io
import math
import json
import base64
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms

from models.jscc_system import DeepJSCCSystem, load_jscc_weights

app = FastAPI(title="Semantic Deep JSCC Backend")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

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
    model = DeepJSCCSystem(use_vae=True).to(device)
    model.eval()
    
    # Trained transmitter/receiver weights; the VAE keeps its pretrained weights.
    # Fail loudly rather than silently streaming output from an untrained model.
    checkpoint_path = 'checkpoints/best_jscc_model.pth'
    epoch = load_jscc_weights(model, checkpoint_path, device)
    print(f"Loaded trained weights from {checkpoint_path} (epoch {epoch + 1}).")

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to video stream.")
    
    try:
        while True:
            # Receive JSON payload from client
            data_str = await websocket.receive_text()
            try:
                out_data_uri, snr_db = process_frame(json.loads(data_str))
            except Exception as e:
                # A bad frame should not kill the stream; reply so the client keeps looping
                print(f"Frame error: {e}")
                await websocket.send_json({"error": str(e)})
                continue

            # Send back the reconstructed frame
            await websocket.send_json({
                "image_reconstructed": out_data_uri,
                "snr": snr_db
            })

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")


def process_frame(data):
    """Decodes one client frame, runs it through the JSCC pipeline, returns (jpeg data URI, snr)."""
    base64_image = data.get("image", "")
    snr_db = float(data.get("snr", 10.0))
    if not math.isfinite(snr_db):
        snr_db = 10.0
    # Stay inside the SNR range the model was trained on
    snr_db = min(max(snr_db, 0.0), 20.0)
    channel_type = data.get("channel", "rician")
    if channel_type not in ['awgn', 'rician', 'rayleigh']:
        channel_type = 'rician'
    
    # Decode base64 image
    if base64_image.startswith("data:image"):
        base64_image = base64_image.split(",")[1]
    
    image_bytes = base64.b64decode(base64_image)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Convert to tensor and add batch dim
    input_tensor = to_tensor(pil_image).unsqueeze(0).to(device)
    
    # Forward pass through Semantic JSCC with CSI
    with torch.no_grad():
        x_hat = model(input_tensor, snr_db=snr_db, channel_type=channel_type)
    
    # Reconstruct to base64
    output_tensor = x_hat.squeeze(0).cpu()
    output_pil = to_pil(output_tensor)
    
    buffer = io.BytesIO()
    output_pil.save(buffer, format="JPEG", quality=85)
    out_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    out_data_uri = f"data:image/jpeg;base64,{out_base64}"
    return out_data_uri, snr_db


# Mount the frontend directory to serve the UI directly
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import os
    if os.path.exists("key.pem") and os.path.exists("cert.pem"):
        print("Starting secure HTTPS server...")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem", reload=True)
    else:
        print("Starting insecure HTTP server...")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
