"""Pinhole camera in the OpenCV convention (x right, y down, z forward).

A world point p maps to camera space as p_c = R p + t, then to pixels as
    u = fx * x/z + cx,  v = fy * y/z + cy.
Pixel (i, j) (column i, row j) has its centre at (i + 0.5, j + 0.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Camera:
    R: torch.Tensor          # (3, 3) world -> camera rotation
    t: torch.Tensor          # (3,)   world -> camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def center(self) -> torch.Tensor:
        """Camera centre in world coordinates: -R^T t."""
        return -self.R.T @ self.t

    @property
    def tan_fovx(self) -> float:
        return self.width / (2 * self.fx)

    @property
    def tan_fovy(self) -> float:
        return self.height / (2 * self.fy)

    def to(self, dtype=None, device=None) -> "Camera":
        return Camera(self.R.to(dtype=dtype, device=device), self.t.to(dtype=dtype, device=device),
                      self.fx, self.fy, self.cx, self.cy, self.width, self.height)


def look_at(eye, target, up=(0.0, 0.0, 1.0), *, width: int, height: int,
            fov_x_deg: float = 60.0, dtype=torch.float32) -> Camera:
    """Camera at `eye` looking at `target`, with `up` roughly pointing up in the image."""
    eye = torch.as_tensor(eye, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    up = torch.as_tensor(up, dtype=torch.float64)
    forward = target - eye
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, up)
    if right.norm() < 1e-8:  # looking straight along `up`
        right = torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    R = torch.stack([right, down, forward], 0)  # rows are the camera axes in world coords
    t = -R @ eye
    fx = width / (2 * math.tan(math.radians(fov_x_deg) / 2))
    return Camera(R.to(dtype), t.to(dtype), fx, fx, width / 2, height / 2, width, height)


def orbit_cameras(n: int, radius: float, *, width: int, height: int, elevation_deg=(20.0, 50.0),
                  fov_x_deg: float = 60.0, target=(0.0, 0.0, 0.0), seed: int = 0,
                  dtype=torch.float32) -> list[Camera]:
    """`n` cameras on a sphere around `target`, evenly spread in azimuth, jittered in elevation."""
    g = torch.Generator().manual_seed(seed)
    cams = []
    lo, hi = elevation_deg
    for k in range(n):
        az = 2 * math.pi * k / n
        el = math.radians(lo + (hi - lo) * torch.rand(1, generator=g).item())
        eye = (target[0] + radius * math.cos(el) * math.cos(az),
               target[1] + radius * math.cos(el) * math.sin(az),
               target[2] + radius * math.sin(el))
        cams.append(look_at(eye, target, width=width, height=height, fov_x_deg=fov_x_deg, dtype=dtype))
    return cams
