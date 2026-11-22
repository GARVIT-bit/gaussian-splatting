"""Fit a single image with 2D Gaussians, reusing the reference compositing.

A warm-up for the 3D case: each Gaussian has a 2D mean (pixels), two log-scales,
a rotation angle, an opacity and an RGB colour. Its covariance is
R(theta) diag(s^2) R(theta)^T; it is blended with exactly the same
front-to-back alpha compositing as the 3D renderer (fixed order, no depth).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import psnr
from .maths import inverse_2x2
from .reference import composite, pixel_centers


@dataclass
class Gaussians2D:
    means: torch.Tensor        # (N, 2) pixels
    log_scales: torch.Tensor   # (N, 2)
    angles: torch.Tensor       # (N,)
    opacity_logits: torch.Tensor
    color_logits: torch.Tensor  # (N, 3), colour = sigmoid

    def params(self):
        return [self.means, self.log_scales, self.angles, self.opacity_logits, self.color_logits]

    @staticmethod
    def random(n: int, height: int, width: int, seed: int = 0, dtype=torch.float32) -> "Gaussians2D":
        g = torch.Generator().manual_seed(seed)
        means = torch.rand(n, 2, generator=g, dtype=dtype) * torch.tensor([width, height], dtype=dtype)
        log_scales = torch.full((n, 2), float(torch.log(torch.tensor(max(height, width) / 10.0))), dtype=dtype)
        angles = torch.rand(n, generator=g, dtype=dtype) * 3.14159
        return Gaussians2D(means, log_scales, angles, torch.zeros(n, dtype=dtype),
                           torch.zeros(n, 3, dtype=dtype))


def render_2d(g: Gaussians2D, height: int, width: int, background: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(g.angles), torch.sin(g.angles)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    M = R * torch.exp(g.log_scales)[:, None, :]
    cov = M @ M.transpose(-1, -2)
    conics, _ = inverse_2x2(cov)
    pix = pixel_centers(height, width, g.means.dtype)
    rgb, _ = composite(pix, g.means, conics, torch.sigmoid(g.opacity_logits),
                       torch.sigmoid(g.color_logits), background)
    return rgb.reshape(height, width, 3)


def fit_image(target: torch.Tensor, n: int = 100, iters: int = 300, lr: float = 0.05, seed: int = 0,
              log_every: int = 0):
    """Fit `target` (H, W, 3) in [0, 1]. Returns (model, rendered image, list of (iter, psnr))."""
    H, W, _ = target.shape
    g = Gaussians2D.random(n, H, W, seed=seed, dtype=target.dtype)
    for p in g.params():
        p.requires_grad_(True)
    lrs = [lr * max(H, W) / 10, lr, lr, lr, lr]  # means move in pixels, the rest in log/logit units
    opt = torch.optim.Adam([{"params": [p], "lr": l} for p, l in zip(g.params(), lrs)])
    bg = torch.zeros(3, dtype=target.dtype)
    history = []
    for it in range(iters + 1):
        img = render_2d(g, H, W, bg)
        if log_every and it % log_every == 0:
            history.append((it, psnr(img.detach(), target)))
        if it == iters:
            break
        loss = (img - target).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return g, img.detach(), history


def synthetic_pattern(height: int, width: int, dtype=torch.float32) -> torch.Tensor:
    """Procedural target: smooth colour gradient with a disc, a square and a stripe."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, height, dtype=dtype), torch.linspace(0, 1, width, dtype=dtype),
                            indexing="ij")
    img = torch.stack([0.2 + 0.6 * xs, 0.3 + 0.4 * ys, 0.6 - 0.4 * xs * ys], -1)
    disc = ((xs - 0.35) ** 2 + (ys - 0.4) ** 2) < 0.05
    img[disc] = torch.tensor([0.95, 0.85, 0.1], dtype=dtype)
    square = (xs > 0.6) & (xs < 0.85) & (ys > 0.55) & (ys < 0.8)
    img[square] = torch.tensor([0.1, 0.2, 0.8], dtype=dtype)
    stripe = (abs(xs + ys - 1.1) < 0.04)
    img[stripe] = torch.tensor([0.9, 0.2, 0.2], dtype=dtype)
    return img
