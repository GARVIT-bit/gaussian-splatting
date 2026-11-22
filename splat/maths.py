"""Core Gaussian maths: quaternion -> rotation, 3D covariance, 2x2 helpers.

Conventions
-----------
* Quaternions are (w, x, y, z) and are normalised inside `quat_to_rotmat`,
  so the optimiser can move them freely.
* Scales are stored as log-scales; the covariance is
      Sigma = R S S^T R^T,   S = diag(exp(log_scale)).
"""
from __future__ import annotations

import torch


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion (w, x, y, z), any norm -> (..., 3, 3) rotation matrix."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return torch.stack(
        [torch.stack([r00, r01, r02], -1),
         torch.stack([r10, r11, r12], -1),
         torch.stack([r20, r21, r22], -1)], -2)


def covariance_3d(log_scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """(N,3) log-scales and (N,4) quaternions -> (N,3,3) covariances R S S^T R^T."""
    R = quat_to_rotmat(quats)
    M = R * torch.exp(log_scales)[..., None, :]  # R @ diag(s): scales the columns
    return M @ M.transpose(-1, -2)


def inverse_2x2(cov: torch.Tensor):
    """(..., 2, 2) symmetric matrix -> (conic (..., 3) = [a, b, c] of the inverse, det).

    The inverse of [[A, B], [B, C]] is [[C, -B], [-B, A]] / det.  We return its
    upper triangle (a, b, c) so that the quadratic form is a dx^2 + 2 b dx dy + c dy^2.
    """
    A = cov[..., 0, 0]
    B = cov[..., 0, 1]
    C = cov[..., 1, 1]
    det = A * C - B * B
    inv_det = 1.0 / det
    conic = torch.stack([C * inv_det, -B * inv_det, A * inv_det], -1)
    return conic, det
