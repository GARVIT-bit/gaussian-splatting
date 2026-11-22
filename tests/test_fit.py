"""Convergence tests: the reference renderer's gradients actually fit images."""
import torch

from splat.camera import orbit_cameras
from splat.fit2d import fit_image, synthetic_pattern
from splat.gaussians import Gaussians
from splat.reference import render
from splat.simple_fit import fit_scene


def test_2d_gaussians_fit_an_image():
    target = synthetic_pattern(32, 32)
    _, img, history = fit_image(target, n=100, iters=200, log_every=50, seed=0)
    first, last = history[0][1], history[-1][1]
    assert last > 28.0, history
    assert last > first + 12.0, history
    psnrs = [p for _, p in history]
    assert psnrs == sorted(psnrs), history  # monotone at this sampling


def test_3d_reference_fits_synthetic_scene_on_held_out_views():
    torch.manual_seed(0)
    gt = Gaussians.random(20, extent=0.7, sh_degree=1, scale_range=(0.05, 0.25), opacity_range=(0.5, 0.95),
                          seed=1000)
    cams = orbit_cameras(16, 3.0, width=24, height=24, fov_x_deg=50.0, seed=0)
    with torch.no_grad():
        imgs = [render(gt, c).image for c in cams]
    test, train = [0, 8], [i for i in range(16) if i not in (0, 8)]
    init = Gaussians.random(100, extent=0.9, sh_degree=1, scale_range=(0.03, 0.08), opacity_range=(0.1, 0.1),
                            seed=0)
    init.sh_dc.zero_()
    init.sh_rest.zero_()
    res = fit_scene(init, [cams[i] for i in train], [imgs[i] for i in train],
                    [cams[i] for i in test], [imgs[i] for i in test], iters=300, seed=0)
    assert res.test_psnr > res.init_test_psnr + 8.0, (res.init_test_psnr, res.test_psnr)
    assert res.test_psnr > 26.0, res.test_psnr
    assert res.test_ssim > 0.85, res.test_ssim
