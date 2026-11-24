"""Triton kernels vs the tiled reference: forward image and every gradient.

Runs on a CUDA GPU, or on the CPU through Triton's interpreter
(TRITON_INTERPRET=1, Linux only; see README "Testing the Triton kernels locally").
Skipped otherwise (e.g. on Windows, where Triton is not installed).

float64: the kernels must match the reference to ~1e-10 (same algorithm, same order).
float32: compared with the float64 reference; tolerances are stated per assert.
"""
import os

import pytest
import torch

triton = pytest.importorskip("triton")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)

from splat.camera import look_at  # noqa: E402
from splat.fast import render_fast  # noqa: E402
from splat.gaussians import Gaussians  # noqa: E402
from splat.reference import render  # noqa: E402
from splat.sh import rgb_to_sh0  # noqa: E402

NAMES = ["means", "log_scales", "quats", "opacity_logits", "sh_dc", "sh_rest"]


def camera(w, h, eye=(0.4, -3.0, 0.7)):
    return look_at(eye, (0.0, 0.0, 0.0), width=w, height=h, fov_x_deg=55.0, dtype=torch.float64)


def scene(n, seed, opacity=(0.2, 0.95), extent=1.0, scales=(0.05, 0.3)):
    g = Gaussians.random(n, extent=extent, sh_degree=3, scale_range=scales, opacity_range=opacity, seed=seed,
                         dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed + 7)
    g.sh_dc[:, 0] = rgb_to_sh0(0.3 + 0.5 * torch.rand(n, 3, generator=gen, dtype=torch.float64))
    g.sh_rest *= 0.4
    return g


def run(g, c, bg, dtype, fast):
    gen = torch.Generator().manual_seed(11)
    up = torch.randn(c.height, c.width, 3, generator=gen, dtype=torch.float64)
    upa = torch.randn(c.height, c.width, generator=gen, dtype=torch.float64)
    dev = DEVICE if fast else "cpu"
    dt = dtype if fast else torch.float64
    ps = {k: v.to(device=dev, dtype=dt).clone().requires_grad_(True) for k, v in g.parameters().items()}
    gg = Gaussians(**ps)
    b = bg.to(device=dev, dtype=dt)
    out = render_fast(gg, c, background=b, backend="triton") if fast else render(gg, c, background=b, mode="tiled")
    ((out.image * up.to(dev, dt)).sum() + 0.5 * (out.alpha * upa.to(dev, dt)).sum()).backward()
    return (out.image.detach().double().cpu(), out.alpha.detach().double().cpu(),
            {k: p.grad.double().cpu() for k, p in ps.items()})


CASES = {
    # name: (n, seed, opacity, extent, scales, width, height)
    "mixed": (60, 0, (0.2, 0.95), 1.0, (0.05, 0.3), 40, 30),
    "opaque_early_stop": (120, 1, (0.95, 0.99), 0.5, (0.2, 0.5), 37, 21),
    "behind_camera_and_offscreen": (80, 2, (0.3, 0.9), 4.0, (0.05, 0.4), 33, 33),
    "huge_gaussians": (6, 3, (0.4, 0.9), 0.3, (2.0, 4.0), 48, 40),
    "sparse_empty_tiles": (4, 4, (0.5, 0.9), 0.4, (0.02, 0.05), 64, 48),
}


@pytest.mark.parametrize("case", list(CASES))
def test_triton_float64_matches_reference_exactly(case):
    n, seed, op, ext, sc, w, h = CASES[case]
    g, c = scene(n, seed, op, ext, sc), camera(w, h)
    bg = torch.tensor([0.3, 0.5, 0.8], dtype=torch.float64)
    img_r, a_r, gr = run(g, c, bg, torch.float64, fast=False)
    img_t, a_t, gt = run(g, c, bg, torch.float64, fast=True)
    assert torch.allclose(img_r, img_t, atol=1e-10)
    assert torch.allclose(a_r, a_t, atol=1e-10)
    for k in NAMES:
        assert torch.allclose(gr[k], gt[k], atol=1e-8, rtol=1e-6), (case, k, (gr[k] - gt[k]).abs().max())


@pytest.mark.parametrize("case", ["mixed", "opaque_early_stop", "huge_gaussians"])
def test_triton_float32_matches_float64_reference(case):
    n, seed, op, ext, sc, w, h = CASES[case]
    g, c = scene(n, seed, op, ext, sc), camera(w, h)
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    img_r, _, gr = run(g, c, bg, torch.float64, fast=False)
    img_t, _, gt = run(g, c, bg, torch.float32, fast=True)
    diff = (img_r - img_t).abs().flatten()
    # float32 rounding; single pixels can jump where alpha sits on the 1/255 cut-off, T on the 1e-4
    # stop, or ceil(3 sigma) flips a radius, so the worst pixel gets a looser bound
    assert diff.mean() < 1e-5, diff.mean()
    assert diff.max() < 1.5e-2, diff.max()
    for k in NAMES:
        rel = (gr[k] - gt[k]).norm() / gr[k].norm().clamp_min(1e-12)
        assert rel < 5e-3, (case, k, rel)


def test_triton_trains_identically_to_autograd_float64():
    """30 training iterations (densification + an opacity reset included) with the Triton kernels
    give the same parameters as autograd through the tiled reference."""
    from tests.test_trainer import small_cfg, tiny_scene
    from splat.trainer import Trainer
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        scene = tiny_scene()
        init = Gaussians.random(40, extent=0.8, sh_degree=3, scale_range=(0.03, 0.08), opacity_range=(0.1, 0.1),
                                seed=3, dtype=torch.float64)
        runs = {}
        for m, dev in (("tiled", "cpu"), ("triton", DEVICE)):
            t = Trainer(scene, init, small_cfg(iterations=30, render_mode=m, device=dev))
            infos = [t.step() for _ in range(30)]
            runs[m] = (t.gaussians, [i.get("densify") for i in infos])
    finally:
        torch.set_default_dtype(old)
    (ga, da), (gb, db) = runs["tiled"], runs["triton"]
    assert da == db and any(d for d in da)
    for k, v in ga.parameters().items():
        assert torch.allclose(v, gb.parameters()[k].cpu(), atol=1e-8), k
