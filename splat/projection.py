"""EWA projection of 3D Gaussians to 2D screen-space Gaussians.

For a Gaussian with world mean mu and covariance Sigma:
    t      = W mu + t_cam                      (camera-space mean, W = world->camera rotation)
    mu_2d  = (fx t_x / t_z + cx, fy t_y / t_z + cy)
    J      = d(mu_2d)/d(t) = [[fx/t_z, 0, -fx t_x/t_z^2],
                              [0, fy/t_z, -fy t_y/t_z^2]]
    Sigma' = J W Sigma W^T J^T + 0.3 I         (first-order / EWA approximation + low-pass)
    conic  = Sigma'^{-1}                       (stored as [a, b, c])
    radius = ceil(3 sqrt(lambda_max(Sigma')))  (3 sigma bound, used for culling / tiles)

As in the reference implementation, t_x/t_z and t_y/t_z are clamped to 1.3x the
half-FoV tangent *only inside J*, so Gaussians far outside the frustum do not
get exploding Jacobians.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .camera import Camera
from .maths import inverse_2x2

LOW_PASS = 0.3
NEAR_PLANE = 0.2
FRUSTUM_SLACK = 1.3


@dataclass
class Projected:
    means2d: torch.Tensor   # (N, 2) pixel coordinates
    depths: torch.Tensor    # (N,)   camera-space z
    cov2d: torch.Tensor     # (N, 2, 2) with the low-pass added
    conics: torch.Tensor    # (N, 3) [a, b, c] of the inverse covariance
    radii: torch.Tensor     # (N,)   integer pixel radius (0 = culled)
    valid: torch.Tensor     # (N,)   bool: in front of the near plane and det > 0


def world_to_camera(means: torch.Tensor, cam: Camera) -> torch.Tensor:
    return means @ cam.R.T + cam.t


def perspective_jacobian(t_cam: torch.Tensor, cam: Camera) -> torch.Tensor:
    """(N, 3) camera-space points -> (N, 2, 3) Jacobian of the pixel projection (with FoV clamp)."""
    x, y, z = t_cam.unbind(-1)
    limx = FRUSTUM_SLACK * cam.tan_fovx
    limy = FRUSTUM_SLACK * cam.tan_fovy
    x = torch.clamp(x / z, -limx, limx) * z
    y = torch.clamp(y / z, -limy, limy) * z
    zero = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([cam.fx / z, zero, -cam.fx * x / (z * z)], -1),
        torch.stack([zero, cam.fy / z, -cam.fy * y / (z * z)], -1),
    ], -2)
    return J


def project_gaussians(means: torch.Tensor, cov3d: torch.Tensor, cam: Camera,
                      near: float = NEAR_PLANE, low_pass: float = LOW_PASS) -> Projected:
    t_cam = world_to_camera(means, cam)
    z = t_cam[:, 2]
    in_front = z > near
    # Gaussians behind the near plane would divide by ~0; give them a harmless depth
    # so no NaN reaches the gradients, and mark them invalid.
    safe_t = torch.where(in_front[:, None], t_cam, torch.ones_like(t_cam))
    x, y, zs = safe_t.unbind(-1)
    means2d = torch.stack([cam.fx * x / zs + cam.cx, cam.fy * y / zs + cam.cy], -1)

    J = perspective_jacobian(safe_t, cam)
    T = J @ cam.R                                   # (N, 2, 3): J W
    cov2d = T @ cov3d @ T.transpose(-1, -2)
    cov2d = cov2d + low_pass * torch.eye(2, dtype=cov2d.dtype, device=cov2d.device)
    conics, det = inverse_2x2(cov2d)
    valid = in_front & (det > 0)

    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lam_max = mid + torch.sqrt(torch.clamp_min(mid * mid - det, 0.1))
    radii = torch.ceil(3.0 * torch.sqrt(lam_max)).detach().to(torch.int64)
    radii = torch.where(valid, radii, torch.zeros_like(radii))
    return Projected(means2d, z, cov2d, conics, radii, valid)
