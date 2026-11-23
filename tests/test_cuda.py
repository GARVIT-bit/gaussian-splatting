"""GPU checks (skipped on the CPU-only laptop; the notebooks run them on the T4).

The tiled reference renderer on CUDA in float32 must match the CPU float64 result,
forward and gradients, within float32 tolerances.
"""
import pytest
import torch

from splat.camera import look_at
from splat.gaussians import Gaussians
from splat.reference import render

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@cuda
def test_tiled_cuda_float32_matches_cpu_float64():
    g64 = Gaussians.random(300, extent=1.0, sh_degree=3, scale_range=(0.02, 0.2), opacity_range=(0.2, 0.9),
                           seed=0, dtype=torch.float64)
    cam = look_at((0.5, -3.5, 1.0), (0.0, 0.0, 0.0), width=100, height=75, fov_x_deg=50.0, dtype=torch.float64)
    bg = torch.tensor([1.0, 1.0, 1.0])
    results = []
    for device, dtype in (("cpu", torch.float64), ("cuda", torch.float32)):
        ps = {k: v.to(device=device, dtype=dtype).clone().requires_grad_(True) for k, v in g64.parameters().items()}
        out = render(Gaussians(**ps), cam, background=bg, mode="tiled")
        w = torch.linspace(0, 1, out.image.numel(), dtype=dtype, device=device).reshape(out.image.shape)
        (out.image * w).sum().backward()
        results.append((out.image.detach().double().cpu(), {k: p.grad.double().cpu() for k, p in ps.items()}))
    (img_a, grad_a), (img_b, grad_b) = results
    # Float32 vs float64 can legitimately disagree where alpha sits right at the 1/255 cut-off
    # or where ceil(3 sigma) flips the radius: those pixels jump by up to ~0.01. So: tight on
    # typical pixels, bounded on the worst one.
    diff = (img_a - img_b).abs().flatten()
    assert diff.mean() < 1e-5
    assert torch.quantile(diff, 0.999) < 1e-4
    assert diff.max() < 1.5e-2
    for k in grad_a:
        rel = (grad_a[k] - grad_b[k]).norm() / grad_a[k].norm().clamp_min(1e-12)
        assert rel < 1e-3, (k, rel)
