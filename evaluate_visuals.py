import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data.dataset import get_cifar10_dataloaders
from models.jscc_system import DeepJSCCSystem
from utils.losses import compute_psnr

def evaluate_snr_curve(model, dataloader, device, snr_list):
    print("Evaluating SNR vs PSNR curve...")
    model.eval()
    psnr_results = []
    
    with torch.no_grad():
        for snr in snr_list:
            total_psnr = 0.0
            batches = 0
            for images, _ in dataloader:
                images = images.to(device)
                x_hat, _, _ = model(images, snr_db=snr, channel_type='rayleigh')
                total_psnr += compute_psnr(x_hat, images).item()
                batches += 1
                
            avg_psnr = total_psnr / batches
            psnr_results.append(avg_psnr)
            print(f"SNR: {snr} dB | Avg PSNR: {avg_psnr:.2f} dB")
            
    plt.figure(figsize=(8, 6))
    plt.plot(snr_list, psnr_results, marker='o', linestyle='-', color='b')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('Performance over Rayleigh Fading Channel')
    plt.savefig('results/psnr_curve.png')
    plt.close()

def generate_visual_grid(model, dataloader, device, low_snr=0.0, high_snr=20.0):
    print("Generating side-by-side reconstruction grids...")
    model.eval()
    with torch.no_grad():
        # Get one batch
        images, _ = next(iter(dataloader))
        # Select the first 8 images
        images = images[:8].to(device)
        
        # High SNR reconstruction
        x_hat_high, _, _ = model(images, snr_db=high_snr, channel_type='rayleigh')
        
        # Low SNR reconstruction
        x_hat_low, _, _ = model(images, snr_db=low_snr, channel_type='rayleigh')
        
        # Create a grid with Row 1 (Originals), Row 2 (High SNR), Row 3 (Low SNR)
        tensor_grid = torch.cat([images, x_hat_high, x_hat_low], dim=0)
        grid_img = make_grid(tensor_grid, nrow=8, padding=2, normalize=False)
        
        plt.figure(figsize=(15, 6))
        # make_grid returns C x H x W, so we permute for matplotlib
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(f"Top: Original | Middle: High SNR ({high_snr}dB) | Bottom: Low SNR ({low_snr}dB)")
        plt.tight_layout()
        plt.savefig('results/reconstruction_grid.png')
        plt.close()

if __name__ == '__main__':
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the DeepJSCCSystem
    model = DeepJSCCSystem().to(device)
    
    # Attempt to load weights
    checkpoint_path = 'checkpoints/best_jscc_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: '{checkpoint_path}' not found. Using untrained weights for evaluation.")
        
    # Get dataloaders
    _, test_loader = get_cifar10_dataloaders(batch_size=64)
    
    # Run evaluation
    snr_list = list(range(0, 22, 2))
    evaluate_snr_curve(model, test_loader, device, snr_list)
    generate_visual_grid(model, test_loader, device, low_snr=0.0, high_snr=20.0)
    
    print("Evaluation complete. Results saved in 'results/' directory.")
