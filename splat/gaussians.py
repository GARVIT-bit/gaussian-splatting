"""Gaussian scene parameterisation.

Every Gaussian stores unconstrained parameters that the optimiser updates:
    means      (N, 3)   world position
    log_scales (N, 3)   scale = exp(log_scale)
    quats      (N, 4)   rotation (w, x, y, z), normalised when used
    opacity_logits (N,) opacity = sigmoid(logit)
    sh_dc      (N, 1, 3)   DC (degree-0) SH coefficient
    sh_rest    (N, K-1, 3) higher-order SH coefficients, K = (max_degree+1)^2
The two SH parts are separate tensors so Adam can give them different learning
rates (Adam is invariant to gradient scaling, so a gradient hook would not do).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .sh import num_sh_coeffs, rgb_to_sh0


@dataclass
class Gaussians:
    means: torch.Tensor
    log_scales: torch.Tensor
    quats: torch.Tensor
    opacity_logits: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    @property
    def sh(self) -> torch.Tensor:
        """(N, K, 3) all SH coefficients."""
        return torch.cat([self.sh_dc, self.sh_rest], 1)

    @property
    def num(self) -> int:
        return self.means.shape[0]

    @property
    def max_sh_degree(self) -> int:
        return int(round((1 + self.sh_rest.shape[1]) ** 0.5)) - 1

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    def parameters(self) -> dict[str, torch.Tensor]:
        return {"means": self.means, "log_scales": self.log_scales, "quats": self.quats,
                "opacity_logits": self.opacity_logits, "sh_dc": self.sh_dc, "sh_rest": self.sh_rest}

    def requires_grad_(self, flag: bool = True) -> "Gaussians":
        for p in self.parameters().values():
            p.requires_grad_(flag)
        return self

    def detach(self) -> "Gaussians":
        return Gaussians(**{k: v.detach().clone() for k, v in self.parameters().items()})

    @staticmethod
    def random(n: int, *, extent: float = 1.0, sh_degree: int = 0, scale_range=(0.03, 0.15),
               opacity_range=(0.3, 0.9), seed: int = 0, dtype=None) -> "Gaussians":
        """Random Gaussians in the box [-extent, extent]^3 (used for synthetic scenes and init)."""
        dtype = dtype or torch.get_default_dtype()
        g = torch.Generator().manual_seed(seed)
        rand = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
        means = (rand(n, 3) * 2 - 1) * extent
        lo, hi = scale_range
        log_scales = torch.log(lo + (hi - lo) * rand(n, 3))
        quats = torch.randn(n, 4, generator=g, dtype=torch.float64)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        olo, ohi = opacity_range
        op = olo + (ohi - olo) * rand(n)
        opacity_logits = torch.log(op / (1 - op))
        sh = torch.zeros(n, num_sh_coeffs(sh_degree), 3, dtype=torch.float64)
        sh[:, 0] = rgb_to_sh0(rand(n, 3))
        if sh_degree > 0:
            sh[:, 1:] = 0.2 * torch.randn(n, num_sh_coeffs(sh_degree) - 1, 3, generator=g, dtype=torch.float64)
        return Gaussians(means.to(dtype), log_scales.to(dtype), quats.to(dtype),
                         opacity_logits.to(dtype), sh[:, :1].to(dtype).contiguous(),
                         sh[:, 1:].to(dtype).contiguous())
