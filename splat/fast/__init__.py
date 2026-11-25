"""Fast rasterizer: hand-written forward + backward, PyTorch spec backend or Triton kernels.

    out = render_fast(gaussians, camera, sh_degree=3, background=bg, backend="triton")

Same semantics as `splat.reference.render(..., mode="tiled")`, which it is tested against.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..camera import Camera
from ..gaussians import Gaussians
from .binning import Bins, bin_gaussians
from .preprocess import PreprocessFn
from .raster_torch import RasterizeTorchFn


@dataclass
class FastProjected:
    means2d: torch.Tensor
    conics: torch.Tensor
    depths: torch.Tensor
    radii: torch.Tensor
    valid: torch.Tensor


@dataclass
class FastOutput:
    image: torch.Tensor
    alpha: torch.Tensor
    projected: FastProjected
    colors: torch.Tensor
    bins: Bins


def render_fast(g: Gaussians, cam: Camera, *, sh_degree: int | None = None, background=None,
                backend: str = "torch", tile: int = 16, batch_elems: int = 1 << 22,
                alpha_max: float | None = None) -> FastOutput:
    """alpha_max: override the 0.99 alpha clamp (Triton backend only; used by the gsplat diagnostic)."""
    dtype, device = g.means.dtype, g.means.device
    if sh_degree is None:
        sh_degree = g.max_sh_degree
    if background is None:
        background = torch.zeros(3, dtype=dtype, device=device)
    background = background.to(dtype=dtype, device=device)
    cam = cam.to(dtype=dtype, device=device)
    means2d, conics, opac, colors, depths, radii, valid = PreprocessFn.apply(
        g.means, g.log_scales, g.quats, g.opacity_logits, g.sh_dc, g.sh_rest, cam, sh_degree)
    bins = bin_gaussians(means2d, radii, depths, valid, cam.width, cam.height, tile)
    if backend == "torch":
        if alpha_max is not None:
            raise ValueError("alpha_max is only supported by the triton backend")
        image, alpha = RasterizeTorchFn.apply(means2d, conics, opac, colors, bins, cam.width, cam.height,
                                              background, batch_elems)
    elif backend == "triton":
        from .raster_triton import RasterizeTritonFn
        from ..reference import ALPHA_MAX
        image, alpha = RasterizeTritonFn.apply(means2d, conics, opac, colors, bins, cam.width, cam.height,
                                               background, ALPHA_MAX if alpha_max is None else alpha_max)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    radii = torch.where(valid, radii, torch.zeros_like(radii))
    return FastOutput(image, alpha, FastProjected(means2d, conics, depths, radii, valid), colors, bins)
