import os
import io
import json
import base64
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms

from models.jscc_system import DeepJSCCSystem

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
    
    # Try to load weights if they exist
    checkpoint_path = 'checkpoints/best_jscc_model.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/latest_jscc_model.pth'

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state'] if (isinstance(checkpoint, dict) and 'model_state' in checkpoint) else checkpoint
            # strict=False: ViT weights load; VAE weights stay at random init (VAE is frozen anyway)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            loaded = len(state_dict) - len(missing)
            print(f"Loaded trained weights from {checkpoint_path} ({loaded} keys matched).")
            if missing:
                print(f"  -> {len(missing)} keys not in checkpoint (VAE stays pretrained — OK).")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}. Using untrained initialization.")
    else:
        print("Warning: No trained weights found. Using untrained initialization.")

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to video stream.")
    
    try:
        while True:
            # Receive JSON payload from client
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            base64_image = data.get("image", "")
            snr_db = float(data.get("snr", 10.0))
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
                # returns (x_hat, s, y)
                x_hat, _, _ = model(input_tensor, snr_db=snr_db, channel_type=channel_type)
            
            # Reconstruct to base64
            output_tensor = x_hat.squeeze(0).cpu()
            output_pil = to_pil(output_tensor)
            
            buffer = io.BytesIO()
            output_pil.save(buffer, format="JPEG", quality=85)
            out_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            out_data_uri = f"data:image/jpeg;base64,{out_base64}"
            
            # Send back the reconstructed frame
            await websocket.send_json({
                "image_reconstructed": out_data_uri,
                "snr": snr_db
            })
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")

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
