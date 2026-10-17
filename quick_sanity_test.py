import torch
import torch.nn.functional as F
from models.jscc_system import DeepJSCCSystem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running Rapid Convergence Sanity Test on: {device}")

# Initialize model with CSI
model = DeepJSCCSystem(use_vae=True).to(device)
model.train()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Create a fixed batch of 8 synthetic image tensors (128x128x3)
torch.manual_seed(42)
fixed_images = torch.rand(8, 3, 128, 128, device=device)

print("\n--- Starting 50-Step Rapid Overfit Test across AWGN, Rician, and Rayleigh ---")
initial_loss = None
final_loss = None
initial_cos = None
final_cos = None

channels = ['awgn', 'rician', 'rayleigh']

for step in range(50):
    optimizer.zero_grad()
    
    # Train across channels and SNR 10-20 dB
    ch = channels[step % 3]
    latents_hat, latents = model.forward_latent_training(fixed_images, snr_db=15.0, channel_type=ch)
    
    mse_loss = F.mse_loss(latents_hat, latents)
    cos_sim = F.cosine_similarity(latents_hat.flatten(1), latents.flatten(1), dim=1).mean()
    loss = mse_loss + 0.1 * (1.0 - cos_sim)
    
    loss.backward()
    optimizer.step()
    
    if step == 0:
        initial_loss = loss.item()
        initial_cos = cos_sim.item()
        print(f"Step  1/50 | Channel: {ch:8s} | Loss: {initial_loss:.4f} | Cosine Similarity: {initial_cos:+.4f}")
    elif (step + 1) % 10 == 0:
        print(f"Step {step+1:2d}/50 | Channel: {ch:8s} | Loss: {loss.item():.4f} | Cosine Similarity: {cos_sim.item():+.4f}")

final_loss = loss.item()
final_cos = cos_sim.item()

print("\n--- Test Results Summary ---")
print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} (Reduction: {(1 - final_loss/initial_loss)*100:.1f}%)")
print(f"Initial Cosine Similarity: {initial_cos:+.4f} -> Final Cosine Similarity: {final_cos:+.4f}")

# Verify that loss decreased significantly and cosine similarity improved dramatically
assert final_loss < initial_loss * 0.3, "Loss did not drop by at least 70%!"
assert final_cos > 0.80, f"Cosine similarity did not reach >0.80! Got {final_cos:.4f}"

print("\n SANITY TEST PASSED: ViT CSI architecture rapidly converges with high directional fidelity!")
