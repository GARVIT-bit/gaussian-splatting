import math

import numpy as np
import pytest
import torch

from splat.camera import orbit_cameras
from splat.data import Frame, Scene
from splat.gaussians import Gaussians
from splat.reference import render
from splat.sh import eval_sh
from splat.trainer import (DensityControl, GaussianOptimizer, TrainConfig, Trainer, expon_lr, init_from_points,
                           inverse_sigmoid, knn_mean_sq_dist)


def test_expon_lr_schedule():
    assert math.isclose(expon_lr(0, 1e-2, 1e-4, 100), 1e-2)
    assert math.isclose(expon_lr(100, 1e-2, 1e-4, 100), 1e-4)
    assert math.isclose(expon_lr(50, 1e-2, 1e-4, 100), 1e-3)          # geometric midpoint
    assert math.isclose(expon_lr(500, 1e-2, 1e-4, 100), 1e-4)         # flat after max_steps
    assert math.isclose(expon_lr(0, 1e-2, 1e-4, 100, delay_steps=10, delay_mult=0.01), 1e-4)
    assert expon_lr(10, 1e-2, 1e-4, 100, 10, 0.01) == pytest.approx(expon_lr(10, 1e-2, 1e-4, 100))


def test_knn_on_a_grid():
    xs = torch.arange(5, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing="ij"), -1).reshape(-1, 3) * 0.5
    d = knn_mean_sq_dist(grid, chunk=7)
    assert torch.allclose(d, torch.full_like(d, 0.25))  # every point has >= 3 neighbours at distance 0.5


def test_init_from_points():
    pts = np.random.default_rng(0).uniform(-1, 1, (50, 3))
    cols = np.random.default_rng(1).uniform(0, 1, (50, 3))
    g = init_from_points(pts, cols, sh_degree=3)
    assert g.sh_rest.shape == (50, 15, 3) and g.max_sh_degree == 3
    assert torch.allclose(g.opacities, torch.full((50,), 0.1))
    dirs = torch.nn.functional.normalize(torch.randn(50, 3), dim=-1)
    assert torch.allclose(eval_sh(3, g.sh, dirs), torch.tensor(cols, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(g.log_scales[:, 0], g.log_scales[:, 2])


def tiny_scene(n_views=6, res=24, seed=0):
    gt = Gaussians.random(25, extent=0.7, sh_degree=1, scale_range=(0.05, 0.25), opacity_range=(0.5, 0.95),
                          seed=1000 + seed)
    cams = orbit_cameras(n_views, 3.0, width=res, height=res, fov_x_deg=50.0, seed=seed)
    frames = []
    for i, c in enumerate(cams):
        with torch.no_grad():
            img = render(gt, c, background=torch.ones(3)).image.clamp(0, 1)
        frames.append(Frame(c, (img * 255).round().to(torch.uint8), f"v{i}"))
    from splat.data import camera_extent
    return Scene(frames[1:], frames[:1], torch.ones(3), camera_extent([f.camera for f in frames[1:]]))


def make_opt(g, extent=1.0, **kw):
    cfg = TrainConfig(**kw)
    return GaussianOptimizer(g, cfg, extent), cfg


def test_optimizer_surgery_keeps_adam_moments_aligned():
    g = Gaussians.random(5, sh_degree=1, seed=0)
    opt, _ = make_opt(g)
    (g.means.sum() * 2 + g.sh.sum() + g.opacities.sum() + g.log_scales.sum() + g.quats.sum()).backward()
    opt.step()
    st = lambda name: opt.adam.state[getattr(opt.g, name)]
    m_before = st("means")["exp_avg"].clone()
    new = {k: v.detach()[:2].clone() for k, v in opt.g.parameters().items()}
    opt.append(new)
    assert opt.g.num == 7
    assert torch.equal(st("means")["exp_avg"][:5], m_before)
    assert (st("means")["exp_avg"][5:] == 0).all() and (st("means")["exp_avg_sq"][5:] == 0).all()
    keep = torch.tensor([True, False, True, True, False, True, True])
    opt.keep(keep)
    assert opt.g.num == 5
    assert torch.equal(st("means")["exp_avg"][:3], m_before[[0, 2, 3]])
    assert st("sh_rest")["exp_avg"].shape == opt.g.sh_rest.shape
    opt.replace("opacity_logits", torch.zeros(5))
    assert (st("opacity_logits")["exp_avg"] == 0).all()
    # the optimiser really updates the new tensors
    before = opt.g.means.detach().clone()
    opt.g.means.sum().backward()
    opt.step()
    assert not torch.equal(before, opt.g.means.detach())
    for group in opt.adam.param_groups:
        assert group["params"][0] is getattr(opt.g, group["name"])


def four_gaussians():
    # 0: small, high gradient -> clone;  1: large, high gradient -> split
    # 2: low opacity -> prune;           3: small, low gradient -> unchanged
    means = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    scales = torch.tensor([[0.005] * 3, [0.08, 0.05, 0.03], [0.005] * 3, [0.005] * 3])  # 1: > 1%, < 10% of extent
    quats = torch.tensor([[1.0, 0, 0, 0], [0.9, 0.1, 0.3, 0.2], [1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    op = torch.tensor([0.5, 0.6, 0.001, 0.5])
    g = Gaussians(means, torch.log(scales), quats, inverse_sigmoid(op), torch.rand(4, 1, 3), torch.rand(4, 3, 3))
    return g


def test_clone_split_prune():
    g = four_gaussians()
    orig = g.detach()
    opt, cfg = make_opt(g)
    dc = DensityControl(4, "cpu")
    dc.grad_accum = torch.tensor([1e-3, 1e-3, 0.0, 1e-5]) * 3
    dc.denom = torch.tensor([3.0, 3.0, 3.0, 3.0])
    info = dc.densify_and_prune(opt, cfg, extent=1.0, max_screen_size=None, generator=torch.Generator().manual_seed(0))
    assert info == {"before": 4, "cloned": 1, "split": 1, "pruned": 1, "low_opacity": 1, "after": 5}
    out = opt.g
    # survivors in order: 0, 3, clone of 0, two children of 1
    assert torch.equal(out.means[0], orig.means[0]) and torch.equal(out.means[1], orig.means[3])
    assert torch.equal(out.means[2], orig.means[0]) and torch.equal(out.sh_rest[2], orig.sh_rest[0])
    child_scales = torch.exp(out.log_scales[3:])
    assert torch.allclose(child_scales, torch.exp(orig.log_scales[1]).expand(2, 3) / 1.6)
    assert torch.equal(out.quats[3], orig.quats[1]) and torch.equal(out.opacity_logits[4], orig.opacity_logits[1])
    off = out.means[3:] - orig.means[1]
    assert (off.norm(dim=-1) < 5 * 0.2).all() and not torch.equal(out.means[3], out.means[4])
    assert dc.grad_accum.shape == (5,) and (dc.denom == 0).all()


def test_screen_and_world_size_pruning_only_with_max_screen_size():
    g = four_gaussians()
    g.log_scales[3] = math.log(0.5)  # > 10% of extent 1.0, low gradient
    opt, cfg = make_opt(g)
    dc = DensityControl(4, "cpu")
    dc.max_radii = torch.tensor([30.0, 0.0, 0.0, 0.0])  # Gaussian 0 got huge on screen
    info = dc.densify_and_prune(opt, cfg, 1.0, None, torch.Generator())
    assert info["after"] == 3  # only the low-opacity one
    g = four_gaussians()
    g.log_scales[3] = math.log(0.5)
    opt, cfg = make_opt(g)
    dc = DensityControl(4, "cpu")
    dc.max_radii = torch.tensor([30.0, 0.0, 0.0, 0.0])
    info = dc.densify_and_prune(opt, cfg, 1.0, 20.0, torch.Generator())
    assert info["after"] == 1 and torch.equal(opt.g.means[0], torch.tensor([1.0, 0, 0]))


def test_gaussian_cap_keeps_highest_gradients():
    g = four_gaussians()
    g.opacity_logits[2] = 0.0  # nothing to prune now
    opt, cfg = make_opt(g, max_gaussians=5)
    dc = DensityControl(4, "cpu")
    dc.grad_accum = torch.tensor([1e-3, 2e-3, 0.0, 0.0])
    dc.denom = torch.ones(4)
    info = dc.densify_and_prune(opt, cfg, 1.0, None, torch.Generator())
    assert info["cloned"] == 0 and info["split"] == 1 and info["after"] == 5


def test_add_stats_uses_ndc_scaled_gradient_norm():
    dc = DensityControl(3, "cpu")
    grad = torch.tensor([[1e-4, 0.0], [0.0, 1e-4], [1.0, 1.0]])
    dc.add_stats(grad, torch.tensor([3, 5, 0]), width=200, height=100)
    assert torch.allclose(dc.grad_accum, torch.tensor([1e-4 * 100, 1e-4 * 50, 0.0]))
    assert torch.equal(dc.denom, torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(dc.max_radii, torch.tensor([3.0, 5.0, 0.0]))


def test_opacity_reset():
    g = four_gaussians()
    opt, _ = make_opt(g)
    DensityControl(4, "cpu").reset_opacity(opt)
    assert torch.allclose(opt.g.opacities, torch.tensor([0.01, 0.01, 0.001, 0.01]), atol=1e-6)


def small_cfg(**kw):
    base = dict(iterations=40, densify_from=4, densify_every=5, densify_until=35, opacity_reset_every=15,
                sh_up_every=10, densify_grad_threshold=1e-5, position_lr_max_steps=40, batch_elems=1 << 16)
    base.update(kw)
    return TrainConfig(**base)


def test_checkpoint_resume_is_bit_identical(tmp_path):
    torch.manual_seed(0)
    scene = tiny_scene()
    init = Gaussians.random(40, extent=0.8, sh_degree=3, scale_range=(0.03, 0.08), opacity_range=(0.1, 0.1), seed=3)
    a = Trainer(scene, init, small_cfg())
    infos = [a.step() for _ in range(40)]
    assert any("densify" in i for i in infos) and any(i.get("opacity_reset") for i in infos)
    assert a.sh_degree == 3

    b = Trainer(scene, init, small_cfg())
    for _ in range(17):
        b.step()
    b.save(str(tmp_path / "ckpt.pt"))
    c = Trainer(scene, Gaussians.random(1, sh_degree=3), small_cfg())
    c.load(str(tmp_path / "ckpt.pt"))
    for _ in range(23):
        c.step()
    assert c.iteration == 40
    for k, v in a.gaussians.parameters().items():
        assert torch.equal(v, c.gaussians.parameters()[k]), k


def test_training_end_to_end_improves_held_out_psnr():
    torch.manual_seed(0)
    scene = tiny_scene(n_views=12, res=32)
    pts = np.random.default_rng(0).uniform(-1.0, 1.0, (300, 3))
    init = init_from_points(pts, np.full((300, 3), 0.5), sh_degree=1)
    cfg = TrainConfig(iterations=400, densify_from=50, densify_every=50, densify_until=300,
                      opacity_reset_every=1000, position_lr_max_steps=400, sh_degree=1,
                      position_lr_init=1.6e-3, position_lr_final=1.6e-5)
    t = Trainer(scene, init, cfg)
    start = t.evaluate()["psnr"]
    infos = [t.step() for _ in range(cfg.iterations)]
    end = t.evaluate()["psnr"]
    assert any("densify" in i for i in infos)
    assert end > start + 6, (start, end)
    assert end > 22, end
