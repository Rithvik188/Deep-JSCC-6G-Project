import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from models.jscc_system import DeepJSCCSystem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = DeepJSCCSystem(use_vae=True).to(device)
model.eval()

checkpoint_path = 'checkpoints/latest_jscc_model.pth'
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    print("Loaded checkpoint successfully.")
else:
    print("Checkpoint not found!")

# Create a test pattern image (gradient / checkerboard)
img = Image.new('RGB', (128, 128), color='red')
to_tensor = transforms.ToTensor()
x = to_tensor(img).unsqueeze(0).to(device)

with torch.no_grad():
    latents = model.vae.encode(x)
    print(f"Latents stats: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}, std={latents.std():.4f}")
    
    s = model.transmitter(latents)
    print(f"Transmitted symbols s stats: shape={s.shape}, mean={s.mean():.4f}, power={(s**2).mean():.4f}")
    
    for snr in [0.0, 10.0, 20.0]:
        y = model.channel(s, snr_db=snr, channel_type='rayleigh')
        latents_hat = model.receiver(y)
        diff = (latents_hat - latents).abs().mean().item()
        print(f"SNR {snr:4.1f} dB -> Latents reconstruction L1 error: {diff:.4f}, latents_hat std={latents_hat.std():.4f}")
        
        x_hat = model.vae.decode(latents_hat)
        print(f"SNR {snr:4.1f} dB -> Decoded image x_hat stats: min={x_hat.min():.4f}, max={x_hat.max():.4f}, mean={x_hat.mean():.4f}")
