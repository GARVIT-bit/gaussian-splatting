"""Pure-PyTorch reference rasterizer (differentiable through autograd).

It is deliberately simple and dense: every pixel looks at every visible Gaussian,
sorted front to back by camera-space depth. This is the ground truth that the
fast tile-based rasterizer (Phase 3) must reproduce, forward and backward.

Per pixel p, with Gaussians sorted by depth (i = 1..N):
    d      = p - mu_2d_i
    power  = -1/2 (a dx^2 + 2 b dx dy + c dy^2)
    alpha_i = min(0.99, o_i exp(power))      skipped if alpha_i < 1/255
    T_i     = prod_{j<i} (1 - alpha_j)       transmittance in front of i
    C       = sum_i alpha_i T_i c_i  +  T_final * background
Compositing stops at the first Gaussian that would push T below 1e-4; that
Gaussian and everything behind it are not blended (same rule as the CUDA code).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .camera import Camera
from .gaussians import Gaussians
from .maths import covariance_3d
from .projection import Projected, project_gaussians
from .sh import eval_sh

ALPHA_MAX = 0.99
ALPHA_MIN = 1.0 / 255.0
T_MIN = 1e-4


@dataclass
class RenderOutput:
    image: torch.Tensor      # (H, W, 3)
    alpha: torch.Tensor      # (H, W) accumulated opacity = 1 - T_final
    projected: Projected
    colors: torch.Tensor     # (N, 3) view-dependent colours from SH
    order: torch.Tensor      # indices of the visible Gaussians, front to back


def pixel_centers(height: int, width: int, dtype, device=None) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(height, dtype=dtype, device=device),
                            torch.arange(width, dtype=dtype, device=device), indexing="ij")
    return torch.stack([xs + 0.5, ys + 0.5], -1).reshape(-1, 2)


def composite(pix: torch.Tensor, means2d: torch.Tensor, conics: torch.Tensor,
              opacities: torch.Tensor, colors: torch.Tensor, background: torch.Tensor):
    """Blend Gaussians (already sorted front to back) at pixel positions `pix` (P, 2).

    Returns (rgb (P, C), accumulated alpha (P,)).
    """
    d = pix[:, None, :] - means2d[None, :, :]                     # (P, N, 2)
    dx, dy = d[..., 0], d[..., 1]
    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy       # (P, N)
    alpha = torch.clamp_max(opacities * torch.exp(power), ALPHA_MAX)
    skip = (power > 0) | (alpha < ALPHA_MIN)
    alpha = torch.where(skip, torch.zeros_like(alpha), alpha)

    # Early termination: keep Gaussian i only while T after it stays >= T_MIN.
    # T is non-increasing along i, so `keep` is a prefix of each row.
    T_after = torch.cumprod(1 - alpha, dim=1)
    keep = T_after >= T_MIN
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1 - alpha, dim=1)
    T_before = torch.cat([torch.ones_like(T_after[:, :1]), T_after[:, :-1]], 1)
    weights = alpha * T_before                                       # (P, N)
    T_final = T_after[:, -1] if alpha.shape[1] > 0 else torch.ones_like(pix[:, 0])
    rgb = weights @ colors + T_final[:, None] * background
    return rgb, 1 - T_final


def render(g: Gaussians, cam: Camera, *, sh_degree: int | None = None,
           background: torch.Tensor | None = None, pixel_chunk: int = 4096) -> RenderOutput:
    dtype, device = g.means.dtype, g.means.device
    if sh_degree is None:
        sh_degree = g.max_sh_degree
    if background is None:
        background = torch.zeros(3, dtype=dtype, device=device)

    cam = cam.to(dtype=dtype, device=device)
    cov3d = covariance_3d(g.log_scales, g.quats)
    proj = project_gaussians(g.means, cov3d, cam)

    dirs = g.means - cam.center
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    colors = eval_sh(sh_degree, g.sh, dirs)

    vis = torch.nonzero(proj.valid).squeeze(-1)
    order = vis[torch.argsort(proj.depths[vis].detach(), stable=True)]

    pix = pixel_centers(cam.height, cam.width, dtype, device)
    m2, cn = proj.means2d[order], proj.conics[order]
    op, col = g.opacities[order], colors[order]
    rgbs, alphas = [], []
    for s in range(0, pix.shape[0], pixel_chunk):
        rgb, acc = composite(pix[s:s + pixel_chunk], m2, cn, op, col, background)
        rgbs.append(rgb)
        alphas.append(acc)
    image = torch.cat(rgbs).reshape(cam.height, cam.width, 3)
    alpha = torch.cat(alphas).reshape(cam.height, cam.width)
    return RenderOutput(image, alpha, proj, colors, order)
