"""Gradient checks of the whole reference renderer in float64 (finite differences vs autograd).

These gradients are the ground truth for the hand-written backward pass of Phase 3.
"""
import pytest
import torch

from splat.camera import look_at
from splat.gaussians import Gaussians
from splat.reference import render
from splat.sh import rgb_to_sh0

pytestmark = pytest.mark.usefixtures("float64")

NAMES = ["means", "log_scales", "quats", "opacity_logits", "sh_dc", "sh_rest"]


def scene(n=5, sh_degree=3, seed=0):
    g = Gaussians.random(n, extent=0.6, sh_degree=sh_degree, scale_range=(0.15, 0.4),
                         opacity_range=(0.3, 0.8), seed=seed, dtype=torch.float64)
    # keep colours away from the clamp at 0, where the gradient is not defined
    gen = torch.Generator().manual_seed(seed + 100)
    g.sh_dc[:, 0] = rgb_to_sh0(0.4 + 0.4 * torch.rand(n, 3, generator=gen))
    g.sh_rest *= 0.3
    return g


def camera(w=8, h=6):
    return look_at((0.5, -3.0, 0.8), (0.0, 0.0, 0.0), width=w, height=h, fov_x_deg=45.0, dtype=torch.float64)


def make_fn(c, sh_degree, bg):
    def f(*params):
        return render(Gaussians(*params), c, sh_degree=sh_degree, background=bg).image
    return f


def test_gradcheck_all_parameters_sh3():
    g = scene()
    c = camera()
    params = [p.clone().requires_grad_(True) for p in g.parameters().values()]
    bg = torch.tensor([0.2, 0.3, 0.1])
    assert torch.autograd.gradcheck(make_fn(c, 3, bg), params, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_gradcheck_overlapping_dense_scene_sh1():
    # more Gaussians, higher opacities: exercises the transmittance products
    g = scene(n=8, sh_degree=1, seed=3)
    g.opacity_logits[:] = 1.5
    c = camera(w=7, h=7)
    params = [p.clone().requires_grad_(True) for p in g.parameters().values()]
    assert torch.autograd.gradcheck(make_fn(c, 1, torch.zeros(3)), params, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_every_parameter_gets_a_nonzero_gradient():
    g = scene()
    c = camera(16, 12)
    params = {k: p.clone().requires_grad_(True) for k, p in g.parameters().items()}
    img = render(Gaussians(**params), c).image
    (img * torch.linspace(0.5, 1.5, img.numel()).reshape(img.shape)).sum().backward()
    for k in NAMES:
        grad = params[k].grad
        assert grad is not None and torch.isfinite(grad).all(), k
        assert grad.abs().sum() > 0, k
