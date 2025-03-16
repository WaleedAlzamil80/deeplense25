import torch
from skimage.metrics import structural_similarity as ssim

def mse_batch(original, upscaled):
    """Compute MSE for a batch."""
    return torch.mean((original - upscaled) ** 2, dim=[1, 2, 3])

def psnr_batch(original, upscaled, data_range=2.0):
    """Compute PSNR for a batch."""
    mse = mse_batch(original, upscaled)
    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-8))  # Avoid division by zero
    return psnr

def ssim_batch(original, upscaled, data_range=2.0):
    """Compute SSIM for a batch (looping needed for now, but still faster)."""
    batch_size = original.shape[0]
    ssim_values = [ssim(original[i, 0].detach().cpu().numpy(), upscaled[i, 0].detach().cpu().numpy(), data_range=data_range) for i in range(batch_size)]
    return torch.tensor(ssim_values, device=original.device)
