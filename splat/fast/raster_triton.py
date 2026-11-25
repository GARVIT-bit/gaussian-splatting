"""Triton tile rasterizer: forward and hand-written backward kernels.

One program per 16x16 tile; its 256 pixels are the lanes of one block.

Forward: walk the tile's depth-sorted Gaussian list front to back. Each pixel
keeps its colour, its transmittance T and `last` (how many list entries it got
to before stopping). The loop ends when every pixel in the tile has stopped.

Backward: walk the same list back to front, starting at the largest `last` in
the tile. Each pixel starts from its final T and reconstructs the T in front of
each Gaussian as T / (1 - alpha), carrying a running sum S of everything behind
the current Gaussian, so that
    dL/dalpha_i = T_i (c_i . g) - S / (1 - alpha_i),  then  S += alpha_i T_i (c_i . g)
Per-Gaussian gradients are summed over the tile's 256 pixels with tl.sum and
added to global memory with one atomic_add per value per tile.
Kernels are dtype-generic (float32 for training; float64 for exact tests).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..reference import ALPHA_MAX, ALPHA_MIN, T_MIN
from .binning import Bins


@triton.jit
def _fwd_kernel(m2_ptr, cn_ptr, op_ptr, col_ptr, gid_ptr, rng_ptr, bg_ptr,
                img_ptr, tf_ptr, last_ptr, W, H, tiles_x,
                TILE: tl.constexpr, ALPHA_MAX: tl.constexpr, ALPHA_MIN: tl.constexpr, T_MIN: tl.constexpr):
    t = tl.program_id(0)
    ty = t // tiles_x
    tx = t % tiles_x
    local = tl.arange(0, TILE * TILE)
    x = tx * TILE + local % TILE
    y = ty * TILE + local // TILE
    inside = (x < W) & (y < H)
    px = x.to(tl.float32) + 0.5
    py = y.to(tl.float32) + 0.5
    start = tl.load(rng_ptr + t)
    end = tl.load(rng_ptr + t + 1)
    bg_r = tl.load(bg_ptr + 0)
    bg_g = tl.load(bg_ptr + 1)
    bg_b = tl.load(bg_ptr + 2)
    zero = tl.zeros([TILE * TILE], dtype=bg_r.dtype)
    T = zero + 1.0
    cr = zero
    cg = zero
    cb = zero
    last = tl.zeros([TILE * TILE], dtype=tl.int32)
    done = ~inside
    j = start
    n_left = tl.sum((~done).to(tl.int32), axis=0)
    while (j < end) & (n_left > 0):
        g = tl.load(gid_ptr + j)
        mx = tl.load(m2_ptr + 2 * g)
        my = tl.load(m2_ptr + 2 * g + 1)
        ca = tl.load(cn_ptr + 3 * g)
        cb_ = tl.load(cn_ptr + 3 * g + 1)
        cc = tl.load(cn_ptr + 3 * g + 2)
        o = tl.load(op_ptr + g)
        dx = px.to(mx.dtype) - mx
        dy = py.to(mx.dtype) - my
        power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb_ * dx * dy
        alpha = tl.minimum(o * tl.exp(power), ALPHA_MAX)
        skip = (power > 0) | (alpha < ALPHA_MIN) | done
        test_T = T * (1 - alpha)
        stop = (~skip) & (test_T < T_MIN)
        done = done | stop
        use = (~skip) & (~stop)
        w = tl.where(use, alpha * T, 0.0)
        cr += w * tl.load(col_ptr + 3 * g)
        cg += w * tl.load(col_ptr + 3 * g + 1)
        cb += w * tl.load(col_ptr + 3 * g + 2)
        T = tl.where(use, test_T, T)
        last = tl.where(use, j - start + 1, last)
        j += 1
        n_left = tl.sum((~done).to(tl.int32), axis=0)
    pix = y * W + x
    tl.store(img_ptr + 3 * pix, cr + T * bg_r, mask=inside)
    tl.store(img_ptr + 3 * pix + 1, cg + T * bg_g, mask=inside)
    tl.store(img_ptr + 3 * pix + 2, cb + T * bg_b, mask=inside)
    tl.store(tf_ptr + pix, T, mask=inside)
    tl.store(last_ptr + pix, last, mask=inside)


@triton.jit
def _bwd_kernel(m2_ptr, cn_ptr, op_ptr, col_ptr, gid_ptr, rng_ptr, bg_ptr,
                tf_ptr, last_ptr, gimg_ptr, gacc_ptr,
                gm2_ptr, gcn_ptr, gop_ptr, gcol_ptr, W, H, tiles_x,
                TILE: tl.constexpr, ALPHA_MAX: tl.constexpr, ALPHA_MIN: tl.constexpr):
    t = tl.program_id(0)
    ty = t // tiles_x
    tx = t % tiles_x
    local = tl.arange(0, TILE * TILE)
    x = tx * TILE + local % TILE
    y = ty * TILE + local // TILE
    inside = (x < W) & (y < H)
    pix = y * W + x
    px = x.to(tl.float32) + 0.5
    py = y.to(tl.float32) + 0.5
    start = tl.load(rng_ptr + t)
    bg_r = tl.load(bg_ptr + 0)
    bg_g = tl.load(bg_ptr + 1)
    bg_b = tl.load(bg_ptr + 2)
    T = tl.load(tf_ptr + pix, mask=inside, other=1.0)
    last = tl.load(last_ptr + pix, mask=inside, other=0)
    gr = tl.load(gimg_ptr + 3 * pix, mask=inside, other=0.0)
    gg = tl.load(gimg_ptr + 3 * pix + 1, mask=inside, other=0.0)
    gb = tl.load(gimg_ptr + 3 * pix + 2, mask=inside, other=0.0)
    ga = tl.load(gacc_ptr + pix, mask=inside, other=0.0)
    S = T * (bg_r * gr + bg_g * gg + bg_b * gb - ga)       # everything behind the current Gaussian
    n = tl.max(last, axis=0)
    for k in range(0, n):
        i = n - 1 - k                                        # position in the tile list
        g = tl.load(gid_ptr + start + i)
        mx = tl.load(m2_ptr + 2 * g)
        my = tl.load(m2_ptr + 2 * g + 1)
        ca = tl.load(cn_ptr + 3 * g)
        cb_ = tl.load(cn_ptr + 3 * g + 1)
        cc = tl.load(cn_ptr + 3 * g + 2)
        o = tl.load(op_ptr + g)
        c_r = tl.load(col_ptr + 3 * g)
        c_g = tl.load(col_ptr + 3 * g + 1)
        c_b = tl.load(col_ptr + 3 * g + 2)
        dx = px.to(mx.dtype) - mx
        dy = py.to(mx.dtype) - my
        power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb_ * dx * dy
        G = tl.exp(power)
        raw = o * G
        alpha = tl.minimum(raw, ALPHA_MAX)
        use = (i < last) & (power <= 0) & (alpha >= ALPHA_MIN) & inside
        one_m = tl.where(use, 1 - alpha, 1.0)
        T_before = T / one_m
        cdotg = c_r * gr + c_g * gg + c_b * gb
        d_alpha = T_before * cdotg - S / one_m
        d_alpha = tl.where(use & (raw <= ALPHA_MAX), d_alpha, 0.0)
        w = tl.where(use, alpha * T_before, 0.0)
        S = S + w * cdotg
        T = T_before
        d_p = d_alpha * o * G
        tl.atomic_add(gcol_ptr + 3 * g, tl.sum(w * gr, axis=0))
        tl.atomic_add(gcol_ptr + 3 * g + 1, tl.sum(w * gg, axis=0))
        tl.atomic_add(gcol_ptr + 3 * g + 2, tl.sum(w * gb, axis=0))
        tl.atomic_add(gop_ptr + g, tl.sum(d_alpha * G, axis=0))
        tl.atomic_add(gm2_ptr + 2 * g, tl.sum(d_p * (ca * dx + cb_ * dy), axis=0))
        tl.atomic_add(gm2_ptr + 2 * g + 1, tl.sum(d_p * (cb_ * dx + cc * dy), axis=0))
        tl.atomic_add(gcn_ptr + 3 * g, tl.sum(d_p * (-0.5 * dx * dx), axis=0))
        tl.atomic_add(gcn_ptr + 3 * g + 1, tl.sum(d_p * (-dx * dy), axis=0))
        tl.atomic_add(gcn_ptr + 3 * g + 2, tl.sum(d_p * (-0.5 * dy * dy), axis=0))


def _c(t: torch.Tensor, dtype) -> torch.Tensor:
    return t.detach().to(dtype).contiguous()


class RasterizeTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, opac, colors, bins: Bins, width: int, height: int, background,
                alpha_max: float = ALPHA_MAX):
        dtype, device = means2d.dtype, means2d.device
        m2, cn, op, col = _c(means2d, dtype), _c(conics, dtype), _c(opac, dtype), _c(colors, dtype)
        bg = _c(background, dtype)
        gid = bins.gauss_ids.to(torch.int32).contiguous()
        rng = bins.ranges.to(torch.int32).contiguous()
        img = torch.empty(height * width * 3, dtype=dtype, device=device)
        tf = torch.empty(height * width, dtype=dtype, device=device)
        last = torch.empty(height * width, dtype=torch.int32, device=device)
        _fwd_kernel[(bins.num_tiles,)](m2, cn, op, col, gid, rng, bg, img, tf, last, width, height, bins.tiles_x,
                                       TILE=bins.tile, ALPHA_MAX=alpha_max, ALPHA_MIN=ALPHA_MIN, T_MIN=T_MIN)
        ctx.save_for_backward(m2, cn, op, col, gid, rng, bg, tf, last)
        ctx.meta = (bins.num_tiles, bins.tiles_x, bins.tile, width, height, alpha_max)
        return img.view(height, width, 3), (1 - tf).view(height, width)

    @staticmethod
    def backward(ctx, g_img, g_acc):
        m2, cn, op, col, gid, rng, bg, tf, last = ctx.saved_tensors
        num_tiles, tiles_x, tile, width, height, alpha_max = ctx.meta
        dtype = m2.dtype
        g_img = torch.zeros(height, width, 3, dtype=dtype, device=m2.device) if g_img is None else g_img
        g_acc = torch.zeros(height, width, dtype=dtype, device=m2.device) if g_acc is None else g_acc
        gm2, gcn, gop, gcol = (torch.zeros_like(t) for t in (m2, cn, op, col))
        _bwd_kernel[(num_tiles,)](m2, cn, op, col, gid, rng, bg, tf, last, _c(g_img, dtype), _c(g_acc, dtype),
                                  gm2, gcn, gop, gcol, width, height, tiles_x,
                                  TILE=tile, ALPHA_MAX=alpha_max, ALPHA_MIN=ALPHA_MIN)
        return gm2, gcn, gop, gcol, None, None, None, None, None
