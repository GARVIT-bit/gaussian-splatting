"""Tile rasterizer in plain PyTorch with a hand-written backward pass.

This is the readable specification of what the Triton kernels compute; it runs
anywhere (CPU, float64) so its gradients can be checked against autograd.

Per pixel, Gaussians i = 1..n of its tile front to back, colours c_i, effective
alphas a_i (0 if skipped / after early stop), T_i = prod_{j<i}(1 - a_j):
    C = sum_i a_i T_i c_i + T_f bg,     acc = 1 - T_f
Gradients, with g = dL/dC and g_acc = dL/dacc:
    dL/dc_i = a_i T_i g
    dL/da_i = T_i (c_i . g) - [ sum_{j>i} a_j T_j (c_j . g) + T_f (bg . g - g_acc) ] / (1 - a_i)
The bracket is everything behind Gaussian i, so a back-to-front pass can keep it
as a running sum (that is what the Triton backward kernel does). Then
    a = min(0.99, o G),  G = exp(p),  p = -1/2 (a' dx^2 + c' dy^2) - b' dx dy,  d = pixel - mu2d
    dL/do = dL/da G,  dL/dp = dL/da o G   (both 0 where o G > 0.99: the clamp is flat there)
    dL/dmu2d = dL/dp (a' dx + b' dy, b' dx + c' dy)
    dL/d(a', b', c') = dL/dp (-dx^2/2, -dx dy, -dy^2/2)
"""
from __future__ import annotations

import torch

from ..reference import ALPHA_MAX, ALPHA_MIN, T_MIN
from .binning import Bins


def _tile_batches(bins: Bins, batch_elems: int):
    counts = bins.ranges[1:] - bins.ranges[:-1]
    nonempty = torch.nonzero(counts).squeeze(-1)
    nonempty = nonempty[torch.argsort(counts[nonempty], stable=True)]
    cl = counts[nonempty].tolist()
    t2 = bins.tile * bins.tile
    i = 0
    while i < len(cl):
        j = i + 1
        while j < len(cl) and (j + 1 - i) * t2 * cl[j] <= batch_elems:
            j += 1
        yield nonempty[i:j]
        i = j


def _gather(bins: Bins, tiles: torch.Tensor, dtype):
    device = bins.ranges.device
    t = bins.tile
    starts = bins.ranges[tiles]
    counts = bins.ranges[tiles + 1] - starts
    k = torch.arange(int(counts.max()), device=device)
    valid = k[None, :] < counts[:, None]
    flat = (starts[:, None] + k[None, :]).clamp_max(bins.gauss_ids.shape[0] - 1)
    idx = torch.where(valid, bins.gauss_ids[flat], torch.zeros_like(flat))
    ty, tx = tiles // bins.tiles_x, tiles % bins.tiles_x
    local = torch.arange(t * t, device=device)
    px = (tx[:, None] * t + local[None, :] % t).to(dtype) + 0.5
    py = (ty[:, None] * t + local[None, :] // t).to(dtype) + 0.5
    rows = ty[:, None] * t + local[None, :] // t
    cols = tx[:, None] * t + local[None, :] % t
    return idx, valid, px, py, rows, cols


def _forward_batch(px, py, m2, cn, op, col, valid):
    dx = px[:, :, None] - m2[:, None, :, 0]
    dy = py[:, :, None] - m2[:, None, :, 1]
    a, b, c = cn[:, None, :, 0], cn[:, None, :, 1], cn[:, None, :, 2]
    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
    G = torch.exp(power)
    raw = op[:, None, :] * G
    alpha = torch.clamp_max(raw, ALPHA_MAX)
    use = (power <= 0) & (alpha >= ALPHA_MIN) & valid[:, None, :]
    alpha = torch.where(use, alpha, 0)
    keep = torch.cumprod(1 - alpha, -1) >= T_MIN
    use = use & keep
    alpha = torch.where(use, alpha, 0)
    T_after = torch.cumprod(1 - alpha, -1)
    T_before = torch.cat([torch.ones_like(T_after[..., :1]), T_after[..., :-1]], -1)
    T_f = T_after[..., -1]
    w = alpha * T_before
    return dict(dx=dx, dy=dy, a=a, b=b, c=c, G=G, raw=raw, alpha=alpha, use=use, T_before=T_before, T_f=T_f, w=w)


class RasterizeTorchFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, opac, colors, bins: Bins, width: int, height: int, background,
                batch_elems: int):
        dtype, device = means2d.dtype, means2d.device
        Ch = colors.shape[-1]
        Hp, Wp = bins.tiles_y * bins.tile, bins.tiles_x * bins.tile
        img = background.to(dtype).expand(Hp * Wp, Ch).clone()
        acc = torch.zeros(Hp * Wp, dtype=dtype, device=device)
        batches = list(_tile_batches(bins, batch_elems))
        for tiles in batches:
            idx, valid, px, py, rows, cols = _gather(bins, tiles, dtype)
            f = _forward_batch(px, py, means2d[idx], conics[idx], opac[idx], colors[idx], valid)
            pix = (rows * Wp + cols).reshape(-1)
            img[pix] = (f["w"] @ colors[idx] + f["T_f"][..., None] * background).reshape(-1, Ch)
            acc[pix] = (1 - f["T_f"]).reshape(-1)
        ctx.save_for_backward(means2d, conics, opac, colors, background)
        ctx.bins, ctx.batches, ctx.Wp, ctx.Hp = bins, batches, Wp, Hp
        ctx.width, ctx.height = width, height
        return img.reshape(Hp, Wp, Ch)[:height, :width], acc.reshape(Hp, Wp)[:height, :width]

    @staticmethod
    def backward(ctx, g_img, g_acc):
        means2d, conics, opac, colors, background = ctx.saved_tensors
        bins, Wp, Hp = ctx.bins, ctx.Wp, ctx.Hp
        dtype = means2d.dtype
        Ch = colors.shape[-1]
        gI = torch.zeros(Hp, Wp, Ch, dtype=dtype, device=means2d.device)
        gA = torch.zeros(Hp, Wp, dtype=dtype, device=means2d.device)
        if g_img is not None:
            gI[:ctx.height, :ctx.width] = g_img
        if g_acc is not None:
            gA[:ctx.height, :ctx.width] = g_acc
        gI, gA = gI.reshape(-1, Ch), gA.reshape(-1)
        g_m2 = torch.zeros_like(means2d)
        g_cn = torch.zeros_like(conics)
        g_op = torch.zeros_like(opac)
        g_col = torch.zeros_like(colors)
        for tiles in ctx.batches:
            idx, valid, px, py, rows, cols = _gather(bins, tiles, dtype)
            col = colors[idx]
            op = opac[idx]
            f = _forward_batch(px, py, means2d[idx], conics[idx], op, col, valid)
            pix = rows * Wp + cols
            g = gI[pix]                                                  # (B, P, C)
            ga = gA[pix]                                                 # (B, P)
            cg = torch.einsum("bkc,bpc->bpk", col, g)                    # c_j . g per pixel
            wcg = f["w"] * cg
            behind = torch.flip(torch.cumsum(torch.flip(wcg, [-1]), -1), [-1]) - wcg   # sum_{j>i}
            behind = behind + (f["T_f"] * ((g * background).sum(-1) - ga))[..., None]
            d_alpha = f["T_before"] * cg - behind / (1 - f["alpha"])
            d_alpha = torch.where(f["use"] & (f["raw"] <= ALPHA_MAX), d_alpha, 0)
            d_col = torch.einsum("bpk,bpc->bkc", f["w"], g)
            d_op = (d_alpha * f["G"]).sum(1)
            d_p = d_alpha * op[:, None, :] * f["G"]
            dx, dy, a, b, c = f["dx"], f["dy"], f["a"], f["b"], f["c"]
            d_mx = (d_p * (a * dx + b * dy)).sum(1)
            d_my = (d_p * (b * dx + c * dy)).sum(1)
            d_a = (d_p * (-0.5 * dx * dx)).sum(1)
            d_b = (d_p * (-dx * dy)).sum(1)
            d_c = (d_p * (-0.5 * dy * dy)).sum(1)
            m = valid
            flat = idx[m]
            g_m2.index_add_(0, flat, torch.stack([d_mx, d_my], -1)[m])
            g_cn.index_add_(0, flat, torch.stack([d_a, d_b, d_c], -1)[m])
            g_op.index_add_(0, flat, d_op[m])
            g_col.index_add_(0, flat, d_col[m])
        return g_m2, g_cn, g_op, g_col, None, None, None, None, None
