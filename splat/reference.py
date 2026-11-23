"""Pure-PyTorch reference rasterizer (differentiable through autograd).

Two modes with the same per-pixel maths:
  * "dense": every pixel looks at every visible Gaussian. Exact, tiny scenes only.
  * "tiled": the image is cut into 16x16 tiles and each tile only looks at the
    Gaussians whose 3-sigma square overlaps it, exactly what the CUDA/Triton
    rasterizers do. This is what training uses and what the fast rasterizer of
    Phase 3 must reproduce, forward and backward.

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
import torch.utils.checkpoint

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
              opacities: torch.Tensor, colors: torch.Tensor, background: torch.Tensor,
              valid: torch.Tensor | None = None):
    """Blend Gaussians (already sorted front to back) at pixel positions.

    Shapes (leading batch dims `...` must match):
        pix (..., P, 2), means2d (..., N, 2), conics (..., N, 3), opacities (..., N),
        colors (..., N, C), background (C,), valid: bool broadcastable to (..., P, N) or None.
    Returns (rgb (..., P, C), accumulated alpha (..., P)).
    """
    d = pix[..., :, None, :] - means2d[..., None, :, :]           # (..., P, N, 2)
    dx, dy = d[..., 0], d[..., 1]
    a = conics[..., None, :, 0]
    b = conics[..., None, :, 1]
    c = conics[..., None, :, 2]
    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy       # (..., P, N)
    alpha = torch.clamp_max(opacities[..., None, :] * torch.exp(power), ALPHA_MAX)
    skip = (power > 0) | (alpha < ALPHA_MIN)
    if valid is not None:
        skip = skip | ~valid
    alpha = torch.where(skip, torch.zeros_like(alpha), alpha)

    # Early termination: keep Gaussian i only while T after it stays >= T_MIN.
    # T is non-increasing along i, so `keep` is a prefix of each row.
    T_after = torch.cumprod(1 - alpha, dim=-1)
    keep = T_after >= T_MIN
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1 - alpha, dim=-1)
    T_before = torch.cat([torch.ones_like(T_after[..., :1]), T_after[..., :-1]], -1)
    weights = alpha * T_before                                       # (..., P, N)
    T_final = T_after[..., -1] if alpha.shape[-1] > 0 else torch.ones_like(pix[..., 0])
    rgb = weights @ colors + T_final[..., None] * background
    return rgb, 1 - T_final


@dataclass
class TileBins:
    """Result of tile binning: for every tile, its Gaussians front to back."""
    tile: int
    tiles_x: int
    tiles_y: int
    tile_ids: torch.Tensor     # (M,) tile of each (tile, Gaussian) pair, sorted by tile then depth
    ranks: torch.Tensor        # (M,) depth rank of the Gaussian (index into the depth-sorted list)
    ranges: torch.Tensor       # (num_tiles + 1,) pairs of tile t are [ranges[t], ranges[t+1])


def bin_gaussians(means2d: torch.Tensor, radii: torch.Tensor, width: int, height: int,
                  tile: int = 16) -> TileBins:
    """Assign depth-sorted Gaussians to the tiles their 3-sigma square overlaps.

    A Gaussian with centre m and radius r touches tile (tx, ty) iff the square
    [m - r, m + r]^2 overlaps the tile's pixel area [tx*T, tx*T+T) x [ty*T, ty*T+T)
    and lies at least partly inside the image. The inputs are already in depth
    order, so a stable sort by tile id keeps each tile's list front to back: the
    same result as sorting the 64-bit key (tile_id << 32 | depth) once.
    """
    device = means2d.device
    m = means2d.detach()
    r = radii.to(m.dtype)
    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    on_screen = ((m[:, 0] + r >= 0) & (m[:, 0] - r < width) & (m[:, 1] + r >= 0)
                 & (m[:, 1] - r < height) & (r > 0))
    x0 = torch.floor((m[:, 0] - r) / tile).clamp(0, tiles_x - 1).long()
    x1 = torch.floor((m[:, 0] + r) / tile).clamp(0, tiles_x - 1).long()
    y0 = torch.floor((m[:, 1] - r) / tile).clamp(0, tiles_y - 1).long()
    y1 = torch.floor((m[:, 1] + r) / tile).clamp(0, tiles_y - 1).long()
    w = x1 - x0 + 1
    cnt = torch.where(on_screen, w * (y1 - y0 + 1), torch.zeros_like(w))
    n = m.shape[0]
    ranks = torch.repeat_interleave(torch.arange(n, device=device), cnt)
    starts = torch.cumsum(cnt, 0) - cnt
    offs = torch.arange(ranks.shape[0], device=device) - starts[ranks]
    tx = x0[ranks] + offs % w[ranks]
    ty = y0[ranks] + offs // w[ranks]
    tile_ids = ty * tiles_x + tx
    perm = torch.argsort(tile_ids, stable=True)
    tile_ids, ranks = tile_ids[perm], ranks[perm]
    counts = torch.bincount(tile_ids, minlength=tiles_x * tiles_y)
    ranges = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(counts, 0)])
    return TileBins(tile, tiles_x, tiles_y, tile_ids, ranks, ranges)


def _composite_tiles(bins: TileBins, tiles: torch.Tensor, m2, cn, op, col, background):
    """Composite a batch of tiles, padding each tile's Gaussian list to the longest one."""
    device, dtype = m2.device, m2.dtype
    t = bins.tile
    starts = bins.ranges[tiles]
    counts = bins.ranges[tiles + 1] - starts
    kmax = int(counts.max())
    k = torch.arange(kmax, device=device)
    valid = k[None, :] < counts[:, None]                              # (B, K)
    flat = (starts[:, None] + k[None, :]).clamp_max(bins.ranks.shape[0] - 1)
    idx = torch.where(valid, bins.ranks[flat], torch.zeros_like(flat))
    ty, tx = tiles // bins.tiles_x, tiles % bins.tiles_x
    local = torch.arange(t * t, device=device)
    px = (tx[:, None] * t + local[None, :] % t).to(dtype) + 0.5
    py = (ty[:, None] * t + local[None, :] // t).to(dtype) + 0.5
    pix = torch.stack([px, py], -1)                                   # (B, t*t, 2)
    return composite(pix, m2[idx], cn[idx], op[idx], col[idx], background, valid[:, None, :])


def rasterize_tiled(means2d, conics, opacities, colors, radii, width: int, height: int,
                    background: torch.Tensor, *, tile: int = 16, batch_elems: int = 1 << 22,
                    checkpoint: bool = True):
    """Tile-based compositing of depth-sorted 2D Gaussians.

    Tiles are processed in batches whose padded size (tiles x pixels x Gaussians)
    stays under `batch_elems`. With `checkpoint`, each batch's intermediates are
    recomputed in the backward pass instead of stored, so memory is about one
    batch rather than the whole image.
    Returns (image (H, W, C), alpha (H, W), bins).
    """
    dtype, device = means2d.dtype, means2d.device
    C = colors.shape[-1]
    bins = bin_gaussians(means2d, radii, width, height, tile)
    counts = bins.ranges[1:] - bins.ranges[:-1]
    nonempty = torch.nonzero(counts).squeeze(-1)
    nonempty = nonempty[torch.argsort(counts[nonempty], stable=True)]  # similar lengths together
    Hp, Wp = bins.tiles_y * tile, bins.tiles_x * tile
    rgb_flat = background.to(dtype).expand(Hp * Wp, C).clone()
    acc_flat = torch.zeros(Hp * Wp, dtype=dtype, device=device)
    counts_ne = counts[nonempty].tolist()
    local = torch.arange(tile * tile, device=device)
    use_ckpt = checkpoint and torch.is_grad_enabled() and any(
        x.requires_grad for x in (means2d, conics, opacities, colors))
    pix_idx, rgb_parts, acc_parts = [], [], []
    i = 0
    while i < len(counts_ne):
        j = i + 1  # counts are sorted ascending, so the last tile of a batch is its longest
        while j < len(counts_ne) and (j + 1 - i) * tile * tile * counts_ne[j] <= batch_elems:
            j += 1
        batch = nonempty[i:j]
        args = (bins, batch, means2d, conics, opacities, colors, background)
        if use_ckpt:
            rgb, acc = torch.utils.checkpoint.checkpoint(_composite_tiles, *args, use_reentrant=False)
        else:
            rgb, acc = _composite_tiles(*args)
        ty, tx = batch // bins.tiles_x, batch % bins.tiles_x
        rows = ty[:, None] * tile + local[None, :] // tile
        cols = tx[:, None] * tile + local[None, :] % tile
        pix_idx.append((rows * Wp + cols).reshape(-1))
        rgb_parts.append(rgb.reshape(-1, C))
        acc_parts.append(acc.reshape(-1))
        i = j
    if pix_idx:
        idx = torch.cat(pix_idx)
        rgb_flat = rgb_flat.index_put((idx,), torch.cat(rgb_parts))
        acc_flat = acc_flat.index_put((idx,), torch.cat(acc_parts))
    image = rgb_flat.reshape(Hp, Wp, C)[:height, :width]
    alpha = acc_flat.reshape(Hp, Wp)[:height, :width]
    return image, alpha, bins


def render(g: Gaussians, cam: Camera, *, sh_degree: int | None = None,
           background: torch.Tensor | None = None, mode: str = "dense", pixel_chunk: int = 4096,
           tile: int = 16, batch_elems: int = 1 << 22) -> RenderOutput:
    """Render `g` from `cam` (see the module docstring for the two modes)."""
    dtype, device = g.means.dtype, g.means.device
    if sh_degree is None:
        sh_degree = g.max_sh_degree
    if background is None:
        background = torch.zeros(3, dtype=dtype, device=device)
    background = background.to(dtype=dtype, device=device)

    cam = cam.to(dtype=dtype, device=device)
    cov3d = covariance_3d(g.log_scales, g.quats)
    proj = project_gaussians(g.means, cov3d, cam)

    dirs = g.means - cam.center
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    colors = eval_sh(sh_degree, g.sh, dirs)

    vis = torch.nonzero(proj.valid).squeeze(-1)
    order = vis[torch.argsort(proj.depths[vis].detach(), stable=True)]

    m2, cn = proj.means2d[order], proj.conics[order]
    op, col = g.opacities[order], colors[order]
    if mode == "tiled":
        image, alpha, _ = rasterize_tiled(m2, cn, op, col, proj.radii[order], cam.width, cam.height,
                                          background, tile=tile, batch_elems=batch_elems)
        return RenderOutput(image, alpha, proj, colors, order)
    if mode != "dense":
        raise ValueError(f"unknown mode {mode!r}")
    pix = pixel_centers(cam.height, cam.width, dtype, device)
    rgbs, alphas = [], []
    for s in range(0, pix.shape[0], pixel_chunk):
        rgb, acc = composite(pix[s:s + pixel_chunk], m2, cn, op, col, background)
        rgbs.append(rgb)
        alphas.append(acc)
    image = torch.cat(rgbs).reshape(cam.height, cam.width, 3)
    alpha = torch.cat(alphas).reshape(cam.height, cam.width)
    return RenderOutput(image, alpha, proj, colors, order)
