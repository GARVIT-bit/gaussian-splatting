import math

import pytest
import torch

from splat.camera import look_at
from splat.gaussians import Gaussians
from splat.maths import covariance_3d
from splat.reference import ALPHA_MAX, ALPHA_MIN, T_MIN, composite, pixel_centers, render
from splat.sh import eval_sh, rgb_to_sh0

pytestmark = pytest.mark.usefixtures("float64")


def cam(width=24, height=18):
    return look_at((0.0, -4.0, 0.0), (0.0, 0.0, 0.0), width=width, height=height,
                   fov_x_deg=40.0, dtype=torch.float64)


def single(mean, scales, quat, opacity, rgb):
    f = lambda v: torch.tensor([v], dtype=torch.float64)
    o = f(opacity)
    return Gaussians(f(mean), torch.log(f(scales)), f(quat),
                     torch.log(o / (1 - o)), rgb_to_sh0(f(rgb))[:, None, :], torch.zeros(1, 0, 3))


def sequential_composite(pix, means2d, conics, opac, colors, bg):
    """Per-pixel loop that mirrors the CUDA forward kernel line by line."""
    out = torch.zeros(pix.shape[0], colors.shape[1])
    for p in range(pix.shape[0]):
        T, C = 1.0, torch.zeros(colors.shape[1])
        for i in range(means2d.shape[0]):
            dx, dy = (pix[p] - means2d[i]).tolist()
            a, b, c = conics[i].tolist()
            power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
            if power > 0:
                continue
            alpha = min(ALPHA_MAX, opac[i].item() * math.exp(power))
            if alpha < ALPHA_MIN:
                continue
            test_T = T * (1 - alpha)
            if test_T < T_MIN:
                break
            C += alpha * T * colors[i]
            T = test_T
        out[p] = C + T * bg
    return out


def test_vectorised_composite_matches_sequential_kernel():
    g = torch.Generator().manual_seed(0)
    n, H, W = 40, 10, 12
    pix = pixel_centers(H, W, torch.float64)
    means2d = torch.rand(n, 2, generator=g) * torch.tensor([W, H])
    A = torch.randn(n, 2, 2, generator=g) * 2
    cov = A @ A.transpose(-1, -2) + 0.5 * torch.eye(2)
    inv = torch.linalg.inv(cov)
    conics = torch.stack([inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]], -1)
    opac = 0.5 + 0.5 * torch.rand(n, generator=g)   # dense enough to hit early termination
    colors = torch.rand(n, 3, generator=g)
    bg = torch.tensor([0.1, 0.2, 0.3])
    fast, _ = composite(pix, means2d, conics, opac, colors, bg)
    slow = sequential_composite(pix, means2d, conics, opac, colors, bg)
    assert torch.allclose(fast, slow, atol=1e-12)


def test_single_gaussian_matches_brute_force_per_pixel():
    """Independent path: autograd Jacobian, torch.linalg.inv, explicit pixel loop."""
    c = cam()
    gs = single([0.1, 0.0, -0.05], [0.3, 0.1, 0.15], [0.9, 0.1, 0.3, -0.2], 0.8, [0.9, 0.4, 0.1])
    out = render(gs, c)

    mean = gs.means[0]
    t = c.R @ mean + c.t
    f = lambda t: torch.stack([c.fx * t[0] / t[2] + c.cx, c.fy * t[1] / t[2] + c.cy])
    J = torch.autograd.functional.jacobian(f, t)
    Sigma = covariance_3d(gs.log_scales, gs.quats)[0]
    cov2 = J @ c.R @ Sigma @ c.R.T @ J.T + 0.3 * torch.eye(2)
    inv = torch.linalg.inv(cov2)
    mu = f(t)
    expected = torch.zeros(c.height, c.width, 3)
    for row in range(c.height):
        for col in range(c.width):
            d = torch.tensor([col + 0.5, row + 0.5]) - mu
            alpha = min(0.99, 0.8 * math.exp(-0.5 * (d @ inv @ d).item()))
            if alpha >= 1 / 255:
                expected[row, col] = alpha * torch.tensor([0.9, 0.4, 0.1])
    assert torch.allclose(out.image, expected, atol=1e-12)
    assert out.image.max() > 0.5  # the Gaussian is actually visible


def test_front_gaussian_occludes_back_and_input_order_does_not_matter():
    c = cam()
    red = single([0.0, -1.0, 0.0], [0.4, 0.4, 0.4], [1, 0, 0, 0], 0.7, [1.0, 0.0, 0.0])
    blue = single([0.0, 1.0, 0.0], [0.4, 0.4, 0.4], [1, 0, 0, 0], 0.7, [0.0, 0.0, 1.0])
    cat = lambda a, b: Gaussians(*[torch.cat([x, y]) for x, y in zip(a.parameters().values(), b.parameters().values())])
    img_rb = render(cat(red, blue), c).image
    img_br = render(cat(blue, red), c).image
    assert torch.allclose(img_rb, img_br)
    centre = img_rb[c.height // 2, c.width // 2]
    # red is nearer: its weight is alpha_r, blue gets (1 - alpha_r) alpha_b (alphas near 0.7 at the centre)
    assert centre[0] > centre[2] > 0
    a_r = render(red, c).alpha[c.height // 2, c.width // 2]
    a_b = render(blue, c).alpha[c.height // 2, c.width // 2]
    assert torch.allclose(centre, torch.tensor([a_r, 0.0, (1 - a_r) * a_b]), atol=1e-12)


def test_alpha_clamped_and_background_blended():
    c = cam()
    # large Gaussian (sigma ~16 px): o * exp(power) > 0.99 at the pixels next to its centre
    gs = single([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1, 0, 0, 0], 1 - 1e-9, [1.0, 1.0, 1.0])
    bg = torch.tensor([0.0, 0.0, 1.0])
    out = render(gs, c, background=bg)
    centre = out.image[c.height // 2, c.width // 2]
    assert torch.allclose(out.alpha.max(), torch.tensor(ALPHA_MAX), atol=1e-6)
    assert torch.allclose(centre, torch.tensor([0.99, 0.99, 0.99 + 0.01]), atol=1e-4)
    empty = render(Gaussians(*(p[:0] for p in gs.parameters().values())), c, background=bg)
    assert torch.allclose(empty.image, bg.expand(c.height, c.width, 3))


def test_early_termination_stops_after_transmittance_drops():
    # 6 identical layers with alpha 0.85 at the centre: T after k layers = 0.15^k,
    # 0.15^4 = 5.1e-4 stays above 1e-4 but 0.15^5 = 7.6e-5 would not, so the 5th
    # layer and everything behind it are not blended.
    pix = torch.tensor([[0.0, 0.0]])
    n = 6
    means2d = torch.zeros(n, 2)
    conics = torch.tensor([[1.0, 0.0, 1.0]] * n)
    opac = torch.full((n,), 0.85)
    colors = torch.eye(n)[:, :n]
    rgb, acc = composite(pix, means2d, conics, opac, colors, torch.zeros(n))
    w = rgb[0]
    expected = torch.tensor([0.85 * 0.15 ** k for k in range(4)] + [0.0, 0.0])
    assert torch.allclose(w, expected, atol=1e-12)
    assert torch.allclose(acc, torch.tensor([1 - 0.15 ** 4]), atol=1e-12)


def test_view_dependent_colour_uses_camera_direction():
    g = Gaussians.random(5, sh_degree=3, seed=1)
    c1 = cam()
    c2 = look_at((4.0, 0.0, 1.0), (0.0, 0.0, 0.0), width=24, height=18, dtype=torch.float64)
    col1, col2 = render(g, c1).colors, render(g, c2).colors
    d1 = g.means - c1.center
    assert torch.allclose(col1, eval_sh(3, g.sh, d1 / d1.norm(dim=-1, keepdim=True)))
    assert not torch.allclose(col1, col2)
