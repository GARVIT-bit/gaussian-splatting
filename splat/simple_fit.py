"""Phase-1 3D fit: optimise Gaussians with the autograd reference renderer.

No density control here (that is Phase 2): a fixed set of Gaussians, random
init in a box, Adam with per-parameter learning rates, (1-l) L1 + l D-SSIM.
Ground truth comes from a known synthetic Gaussian scene rendered from orbit
cameras, so the result can be checked on held-out views.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .camera import Camera
from .gaussians import Gaussians
from .losses import gs_loss, psnr, ssim
from .reference import render

DEFAULT_LRS = {"means": 0.004, "log_scales": 0.01, "quats": 0.005, "opacity_logits": 0.05,
               "sh_dc": 0.01, "sh_rest": 0.0005}


@dataclass
class FitResult:
    model: Gaussians
    history: list = field(default_factory=list)   # (iter, loss, mean test psnr)
    test_psnr: float = 0.0
    test_ssim: float = 0.0
    init_test_psnr: float = 0.0


def make_optimizer(g: Gaussians, lrs: dict) -> torch.optim.Adam:
    return torch.optim.Adam([
        {"params": [g.means], "lr": lrs["means"]},
        {"params": [g.log_scales], "lr": lrs["log_scales"]},
        {"params": [g.quats], "lr": lrs["quats"]},
        {"params": [g.opacity_logits], "lr": lrs["opacity_logits"]},
        {"params": [g.sh_dc], "lr": lrs["sh_dc"]},
        {"params": [g.sh_rest], "lr": lrs["sh_rest"]},
    ], eps=1e-15)


@torch.no_grad()
def evaluate(g: Gaussians, cams: list[Camera], images: list[torch.Tensor], background) -> tuple[float, float]:
    ps, ss = [], []
    for c, gt in zip(cams, images):
        img = render(g, c, background=background).image.clamp(0, 1)
        ps.append(psnr(img, gt))
        ss.append(ssim(img, gt).item())
    return sum(ps) / len(ps), sum(ss) / len(ss)


def fit_scene(init: Gaussians, train_cams, train_images, test_cams, test_images, *, iters: int,
              lrs: dict | None = None, lambda_dssim: float = 0.2, background=None, seed: int = 0,
              eval_every: int = 0, log=None) -> FitResult:
    """Fit `init` to the training views; Adam eps = 1e-15 as in the 3DGS code."""
    lrs = {**DEFAULT_LRS, **(lrs or {})}
    if background is None:
        background = torch.zeros(3, dtype=init.means.dtype)
    g = init.detach().requires_grad_(True)
    opt = make_optimizer(g, lrs)
    rng = random.Random(seed)
    res = FitResult(model=g)
    res.init_test_psnr, _ = evaluate(g, test_cams, test_images, background)
    order: list[int] = []
    for it in range(1, iters + 1):
        if not order:
            order = list(range(len(train_cams)))
            rng.shuffle(order)
        k = order.pop()
        img = render(g, train_cams[k], background=background).image
        loss = gs_loss(img, train_images[k], lambda_dssim)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if eval_every and (it % eval_every == 0 or it == iters):
            p, _ = evaluate(g, test_cams, test_images, background)
            res.history.append((it, loss.item(), p))
            if log:
                log(f"iter {it:5d}  loss {loss.item():.4f}  test PSNR {p:.2f} dB")
    res.model = g.detach()
    res.test_psnr, res.test_ssim = evaluate(res.model, test_cams, test_images, background)
    return res
