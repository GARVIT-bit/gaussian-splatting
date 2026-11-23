"""Tiled reference mode: binning rule, equality with the dense mode, gradients."""
import pytest
import torch

from splat.camera import look_at
from splat.gaussians import Gaussians
from splat.reference import bin_gaussians, composite, pixel_centers, rasterize_tiled, render
from splat.sh import rgb_to_sh0

pytestmark = pytest.mark.usefixtures("float64")


def camera(w=40, h=27):  # deliberately not multiples of 16
    return look_at((0.3, -3.0, 0.6), (0.0, 0.0, 0.0), width=w, height=h, fov_x_deg=50.0, dtype=torch.float64)


def scene(n=60, seed=0, opacity=(0.1, 0.3), extent=1.2):
    # extent 1.2 puts some Gaussians partly or fully off-screen
    return Gaussians.random(n, extent=extent, sh_degree=1, scale_range=(0.03, 0.3), opacity_range=opacity,
                            seed=seed, dtype=torch.float64)


def test_binning_matches_brute_force_overlap_rule():
    g = torch.Generator().manual_seed(0)
    n, W, H, T = 50, 70, 45, 16
    m = torch.rand(n, 2, generator=g) * torch.tensor([W + 40.0, H + 40.0]) - 20
    r = torch.randint(0, 25, (n,), generator=g)
    bins = bin_gaussians(m, r, W, H, T)
    for t in range(bins.tiles_x * bins.tiles_y):
        tx, ty = t % bins.tiles_x, t // bins.tiles_x
        got = bins.ranks[bins.ranges[t]:bins.ranges[t + 1]].tolist()
        want = []
        for i in range(n):
            x, y, rr = m[i, 0].item(), m[i, 1].item(), r[i].item()
            if rr == 0 or x + rr < 0 or x - rr >= W or y + rr < 0 or y - rr >= H:
                continue
            # square [x-r, x+r] overlaps tile columns [tx*T, tx*T+T) (clamped to the image)
            if max(x - rr, 0) < tx * T + T and min(x + rr, W - 1e-9) >= tx * T and \
               max(y - rr, 0) < ty * T + T and min(y + rr, H - 1e-9) >= ty * T:
                want.append(i)
        assert got == want, (t, got, want)


def test_tiled_equals_dense_for_low_opacity():
    # with opacity < 0.35, alpha outside the 3-sigma square is < 0.35 * e^-4.5 < 1/255,
    # so culling by tile cannot change any pixel
    c = camera()
    g = scene()
    bg = torch.tensor([0.3, 0.6, 0.9])
    dense = render(g, c, background=bg, mode="dense")
    tiled = render(g, c, background=bg, mode="tiled")
    assert dense.image.abs().sum() > 1.0
    assert torch.allclose(dense.image, tiled.image, atol=1e-12)
    assert torch.allclose(dense.alpha, tiled.alpha, atol=1e-12)


def test_tiled_equals_dense_with_explicit_tile_mask_for_high_opacity():
    c = camera(128, 96)  # 8x6 tiles, so most Gaussians do not touch every tile
    g = scene(n=40, opacity=(0.9, 0.99), seed=1)
    bg = torch.tensor([1.0, 1.0, 1.0])
    out = render(g, c, background=bg, mode="tiled")
    proj = out.projected
    m2 = proj.means2d[out.order].detach()
    r = proj.radii[out.order]
    bins = bin_gaussians(m2, r, c.width, c.height)
    pix = pixel_centers(c.height, c.width, torch.float64)
    tile_of_pix = (pix[:, 1] // 16).long() * bins.tiles_x + (pix[:, 0] // 16).long()
    mask = torch.zeros(pix.shape[0], m2.shape[0], dtype=torch.bool)
    for t in range(bins.tiles_x * bins.tiles_y):
        ranks = bins.ranks[bins.ranges[t]:bins.ranges[t + 1]]
        rows = torch.nonzero(tile_of_pix == t).squeeze(-1)
        mask[rows[:, None], ranks[None, :]] = True
    rgb, _ = composite(pix, proj.means2d[out.order], proj.conics[out.order], g.opacities[out.order],
                       out.colors[out.order], bg, mask)
    assert torch.allclose(out.image, rgb.reshape(c.height, c.width, 3), atol=1e-12)
    # culling does make a difference here (so the test is not vacuous), and it is small:
    # outside the 3-sigma square alpha <= 0.99 * exp(-4.5) ~ 0.011
    dense = render(g, c, background=bg, mode="dense").image
    diff = (dense - out.image).abs().max()
    assert 1e-6 < diff < 0.02, diff


def test_batching_does_not_change_the_result():
    c = camera()
    g = scene(seed=2, opacity=(0.3, 0.9))
    big = render(g, c, mode="tiled", batch_elems=1 << 30).image
    small = render(g, c, mode="tiled", batch_elems=1).image  # one tile per batch
    assert torch.allclose(big, small, atol=1e-12)


def test_empty_and_offscreen_scenes_give_background():
    c = camera()
    bg = torch.tensor([0.2, 0.4, 0.6])
    g = scene(n=5)
    g.means = g.means + torch.tensor([100.0, 0.0, 0.0])   # far off to the side
    out = render(g, c, background=bg, mode="tiled")
    assert torch.allclose(out.image, bg.expand_as(out.image))
    empty = Gaussians(*(p[:0] for p in g.parameters().values()))
    out = render(empty, c, background=bg, mode="tiled")
    assert torch.allclose(out.image, bg.expand_as(out.image))
    # rasterize_tiled with no Gaussians at all
    img, alpha, _ = rasterize_tiled(torch.zeros(0, 2), torch.zeros(0, 3), torch.zeros(0), torch.zeros(0, 3),
                                    torch.zeros(0, dtype=torch.long), 20, 10, bg)
    assert torch.allclose(img, bg.expand(10, 20, 3)) and (alpha == 0).all()


def test_gradcheck_tiled_mode():
    c = camera(w=20, h=18)
    g = Gaussians.random(6, extent=0.6, sh_degree=1, scale_range=(0.1, 0.3), opacity_range=(0.3, 0.8), seed=4,
                         dtype=torch.float64)
    g.sh_dc[:, 0] = rgb_to_sh0(0.4 + 0.4 * torch.rand(6, 3, generator=torch.Generator().manual_seed(5),
                                                       dtype=torch.float64))
    g.sh_rest *= 0.3
    params = [p.clone().requires_grad_(True) for p in g.parameters().values()]
    bg = torch.tensor([0.5, 0.5, 0.5])

    def f(*ps):
        return render(Gaussians(*ps), c, background=bg, mode="tiled", batch_elems=256 * 4).image

    assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_checkpointing_gives_identical_gradients():
    c = camera()
    g = scene(seed=6, opacity=(0.3, 0.9))
    grads = []
    for ckpt in (True, False):
        ps = {k: v.clone().requires_grad_(True) for k, v in g.parameters().items()}
        gg = Gaussians(**ps)
        from splat.maths import covariance_3d
        from splat.projection import project_gaussians
        from splat.sh import eval_sh
        proj = project_gaussians(gg.means, covariance_3d(gg.log_scales, gg.quats), c)
        d = gg.means - c.center
        col = eval_sh(1, gg.sh, d / d.norm(dim=-1, keepdim=True))
        order = torch.nonzero(proj.valid).squeeze(-1)
        order = order[torch.argsort(proj.depths[order])]
        img, _, _ = rasterize_tiled(proj.means2d[order], proj.conics[order], gg.opacities[order], col[order],
                                    proj.radii[order], c.width, c.height, torch.zeros(3), checkpoint=ckpt,
                                    batch_elems=256 * 8)
        (img * torch.linspace(0, 1, img.numel()).reshape(img.shape)).sum().backward()
        grads.append([p.grad.clone() for p in ps.values()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-12)
