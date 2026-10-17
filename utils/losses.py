import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

def compute_psnr(x_hat, x, max_val=1.0):
    mse = F.mse_loss(x_hat, x)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10((max_val ** 2) / mse)

class JSCCLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha

    def forward(self, x_hat, x):
        mse_loss = F.mse_loss(x_hat, x)
        try:
            # ms_ssim takes inputs in [0, 1]
            msssim_loss = 1 - ms_ssim(x_hat, x, data_range=1.0, size_average=True)
            return self.alpha * msssim_loss + (1 - self.alpha) * mse_loss
        except Exception:
            # Fallback to MSE if image size is too small for MS-SSIM
            return mse_loss
