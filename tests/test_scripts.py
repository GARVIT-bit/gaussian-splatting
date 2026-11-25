"""train.py end to end (JSON summary), the trainer's render_fn hook, and the gsplat adapter used by
bench/train_gsplat.py (driven by a fake `gsplat` module, so gsplat itself is not needed)."""
import json
import os
import sys
import types

import torch

from splat.camera import orbit_cameras
from splat.gaussians import Gaussians
from splat.reference import render
from splat.trainer import Trainer
from tests.test_data import write_blender_dataset
from tests.test_trainer import small_cfg, tiny_scene

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "bench"))


def tiny_blender(tmp_path):
    gt = Gaussians.random(30, extent=0.7, sh_degree=1, scale_range=(0.05, 0.2), opacity_range=(0.5, 0.9), seed=1)
    cams = orbit_cameras(6, 3.0, width=24, height=24, fov_x_deg=45.0, seed=0)
    write_blender_dataset(str(tmp_path / "scene"), {"train": cams[:5], "test": cams[5:]}, gt, 45.0)
    return str(tmp_path / "scene")


def test_train_py_writes_results_md_and_json(tmp_path):
    import train
    data = tiny_blender(tmp_path)
    out_md = tmp_path / "res" / "tiny.md"
    train.main(["--data", data, "--format", "blender", "--init-points", "200", "--out", str(tmp_path / "run"),
                "--name", "tiny", "--results", str(out_md), "--iterations", "6", "--log-every", "3",
                "--ckpt-every", "3", "--device", "cpu", "--n-renders", "1"])
    s = json.loads((tmp_path / "res" / "tiny.json").read_text())
    for k in ("psnr", "ssim", "gaussians", "train_hours", "render_ms", "commit", "width", "n_test", "config"):
        assert k in s, k
    assert s["iterations"] == 6 and s["n_test"] == 1 and s["width"] == 24
    assert out_md.exists() and (tmp_path / "run" / "final.pt").exists()


def test_render_fn_hook_trains_like_the_builtin_renderer():
    scene = tiny_scene()
    init = Gaussians.random(40, extent=0.8, sh_degree=3, scale_range=(0.03, 0.08), opacity_range=(0.1, 0.1), seed=3)
    hook = lambda g, cam, *, sh_degree, background: render(g, cam, sh_degree=sh_degree, background=background,
                                                          mode="tiled")
    a = Trainer(scene, init, small_cfg(iterations=20))
    b = Trainer(scene, init, small_cfg(iterations=20), render_fn=hook)
    for _ in range(20):
        a.step()
        b.step()
    for k, v in a.gaussians.parameters().items():
        assert torch.equal(v, b.gaussians.parameters()[k]), k


def fake_gsplat():
    """A stand-in for gsplat.rasterization with gsplat 1.5's output shapes, built on my reference."""
    mod = types.ModuleType("gsplat")
    mod.__version__ = "fake"

    def rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree=None,
                      backgrounds=None, **kw):
        from splat.camera import Camera
        from splat.maths import covariance_3d
        from splat.projection import project_gaussians
        g = Gaussians(means, torch.log(scales), quats, torch.logit(opacities), colors[:, :1], colors[:, 1:])
        K = Ks[0]
        cam = Camera(viewmats[0, :3, :3], viewmats[0, :3, 3], K[0, 0].item(), K[1, 1].item(), K[0, 2].item(),
                     K[1, 2].item(), width, height)
        # like gsplat, the returned (1, N, 2) means2d must be the tensor the image is rasterized from
        from splat.reference import rasterize_tiled
        from splat.sh import eval_sh
        cam = cam.to(dtype=means.dtype)
        proj = project_gaussians(g.means, covariance_3d(g.log_scales, g.quats), cam)
        means2d = proj.means2d[None]
        d = g.means - cam.center
        col = eval_sh(sh_degree, g.sh, d / d.norm(dim=-1, keepdim=True))
        vis = torch.nonzero(proj.valid).squeeze(-1)
        order = vis[torch.argsort(proj.depths[vis].detach().to(torch.float32), stable=True)]
        img, alpha, _ = rasterize_tiled(means2d[0][order], proj.conics[order], g.opacities[order], col[order],
                                        proj.radii[order], width, height, backgrounds[0])
        r = proj.radii
        return img[None], alpha[None, ..., None], {"means2d": means2d, "radii": torch.stack([r, r], -1)[None]}
    mod.rasterization = rasterization
    return mod


def test_gsplat_adapter_trains_like_my_reference(monkeypatch):
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat())
    import train_gsplat
    scene = tiny_scene()
    init = Gaussians.random(40, extent=0.8, sh_degree=3, scale_range=(0.03, 0.08), opacity_range=(0.1, 0.1), seed=3)
    a = Trainer(scene, init, small_cfg(iterations=20))
    b = Trainer(scene, init, small_cfg(iterations=20), render_fn=train_gsplat.gsplat_render)
    ia = [a.step() for _ in range(20)]
    ib = [b.step() for _ in range(20)]
    assert [i.get("densify") for i in ia] == [i.get("densify") for i in ib]
    assert any(i.get("densify") for i in ia)
    for k, v in a.gaussians.parameters().items():
        assert torch.allclose(v, b.gaussians.parameters()[k], atol=1e-6), k


def test_phase4_tables_layout_and_labels():
    import phase4_tables
    paper = json.load(open(os.path.join(ROOT, "bench", "paper_numbers.json"), encoding="utf-8"))
    run = lambda scene, renderer, psnr: {"data": scene, "renderer": renderer, "psnr": psnr, "ssim": 0.9, "lpips": 0.1,
                                         "width": 800, "height": 800, "gaussians": 1000, "train_hours": 1.0,
                                         "render_ms": 10.0}
    lines = phase4_tables.build([run("lego", "triton", 35.0), run("lego", "gsplat 1.5.3 rasterization", 35.2),
                                 run("garden", "triton", 27.0)], paper)
    text = "\n".join(lines)
    assert "reported" in text and "not measured here" in text
    lego = next(l for l in lines if l.startswith("| lego |"))
    assert "35.00" in lego and "35.20" in lego and "35.78" in lego      # mine, gsplat, paper (Table 2)
    garden = next(l for l in lines if l.startswith("| garden |"))
    assert "27.00" in garden and "27.41" in garden and garden.count("–") >= 3   # no gsplat run yet
