"""Tile binning with one sort of 64-bit keys, as in the CUDA rasterizer.

Every (Gaussian, tile) pair gets the key  (tile_id << 32) | float_bits(depth).
Depths are > the near plane, so positive, and the IEEE-754 bit pattern of a
positive float32 increases with its value: sorting the int64 keys once orders
the pairs by tile, and within a tile front to back.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Bins:
    tile: int
    tiles_x: int
    tiles_y: int
    gauss_ids: torch.Tensor    # (M,) Gaussian index of each pair, sorted by (tile, depth)
    tile_ids: torch.Tensor     # (M,)
    ranges: torch.Tensor       # (num_tiles + 1,) pairs of tile t are [ranges[t], ranges[t+1])

    @property
    def num_tiles(self) -> int:
        return self.tiles_x * self.tiles_y


def tile_rects(means2d, radii, width, height, tile):
    """Inclusive tile rectangle [x0, x1] x [y0, y1] touched by each Gaussian's 3-sigma square,
    and how many tiles that is (0 if the square misses the image or radius is 0)."""
    m = means2d.detach()
    r = radii.to(m.dtype)
    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    on = (m[:, 0] + r >= 0) & (m[:, 0] - r < width) & (m[:, 1] + r >= 0) & (m[:, 1] - r < height) & (radii > 0)
    x0 = torch.floor((m[:, 0] - r) / tile).clamp(0, tiles_x - 1).long()
    x1 = torch.floor((m[:, 0] + r) / tile).clamp(0, tiles_x - 1).long()
    y0 = torch.floor((m[:, 1] - r) / tile).clamp(0, tiles_y - 1).long()
    y1 = torch.floor((m[:, 1] + r) / tile).clamp(0, tiles_y - 1).long()
    cnt = torch.where(on, (x1 - x0 + 1) * (y1 - y0 + 1), torch.zeros_like(x0))
    return x0, x1, y0, cnt, tiles_x, tiles_y


def depth_bits(depths: torch.Tensor) -> torch.Tensor:
    """Monotone int64 code of positive depths (bit pattern of their float32 value)."""
    return depths.detach().to(torch.float32).contiguous().view(torch.int32).to(torch.int64)


def bin_gaussians(means2d, radii, depths, valid, width: int, height: int, tile: int = 16) -> Bins:
    device = means2d.device
    radii = torch.where(valid, radii, torch.zeros_like(radii))
    x0, x1, y0, cnt, tiles_x, tiles_y = tile_rects(means2d, radii, width, height, tile)
    n = means2d.shape[0]
    gid = torch.repeat_interleave(torch.arange(n, device=device), cnt)
    starts = torch.cumsum(cnt, 0) - cnt
    offs = torch.arange(gid.shape[0], device=device) - starts[gid]
    w = (x1 - x0 + 1)[gid]
    tile_ids = (y0[gid] + offs // w) * tiles_x + (x0[gid] + offs % w)
    keys = (tile_ids << 32) | depth_bits(depths)[gid]
    keys, perm = torch.sort(keys, stable=True)
    gid = gid[perm]
    tile_ids = keys >> 32
    counts = torch.bincount(tile_ids, minlength=tiles_x * tiles_y)
    ranges = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(counts, 0)])
    return Bins(tile, tiles_x, tiles_y, gid, tile_ids, ranges)
