"""Real spherical harmonics up to degree 3, in the convention of the 3DGS paper.

Basis order per degree l is m = -l..l, real SH with the Condon-Shortley phase.
Colour = sum_k c_k Y_k(d) + 0.5, clamped at 0 (the +0.5 offset lets a zero-
initialised DC term mean mid-grey, as in the original implementation).
"""
from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435)


def num_sh_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """(..., 3) unit directions -> (..., (degree+1)^2) basis values."""
    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, C0)]
    if degree >= 1:
        out += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        out += [C2[0] * xy,
                C2[1] * yz,
                C2[2] * (2 * zz - xx - yy),
                C2[3] * xz,
                C2[4] * (xx - yy)]
    if degree >= 3:
        out += [C3[0] * y * (3 * xx - yy),
                C3[1] * xy * z,
                C3[2] * y * (4 * zz - xx - yy),
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                C3[4] * x * (4 * zz - xx - yy),
                C3[5] * z * (xx - yy),
                C3[6] * x * (xx - 3 * yy)]
    return torch.stack(out, -1)


def eval_sh(degree: int, coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """coeffs (N, K, 3) with K >= (degree+1)^2, dirs (N, 3) unit -> RGB (N, 3) >= 0.

    Only the first (degree+1)^2 coefficients are used, so training can raise the
    active degree without reallocating parameters.
    """
    k = num_sh_coeffs(degree)
    basis = sh_basis(degree, dirs)                      # (N, k)
    rgb = (basis[..., None] * coeffs[:, :k]).sum(-2)    # (N, 3)
    return torch.clamp_min(rgb + 0.5, 0.0)


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """DC coefficient that reproduces `rgb` (inverse of the degree-0 evaluation)."""
    return (rgb - 0.5) / C0
