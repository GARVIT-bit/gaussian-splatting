"""Image losses and metrics: L1, SSIM (11x11 Gaussian window, sigma 1.5), PSNR."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, channels: int, dtype, device) -> torch.Tensor:
    x = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.expand(channels, 1, size, size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM of two (H, W, C) images in [0, 1] (zero-padded borders, as in the 3DGS code)."""
    x = img1.permute(2, 0, 1)[None]
    y = img2.permute(2, 0, 1)[None]
    C = x.shape[1]
    w = _gaussian_window(window_size, sigma, C, x.dtype, x.device)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=C)
    mu_y = F.conv2d(y, w, padding=pad, groups=C)
    sxx = F.conv2d(x * x, w, padding=pad, groups=C) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=C) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=C) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return (num / den).mean()


def l1(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return (img1 - img2).abs().mean()


def gs_loss(pred: torch.Tensor, gt: torch.Tensor, lambda_dssim: float = 0.2) -> torch.Tensor:
    """(1 - lambda) L1 + lambda (1 - SSIM), the 3DGS training loss."""
    return (1 - lambda_dssim) * l1(pred, gt) + lambda_dssim * (1 - ssim(pred, gt))


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = ((img1 - img2) ** 2).mean().item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)
