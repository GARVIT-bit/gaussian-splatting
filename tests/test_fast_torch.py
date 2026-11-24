"""Hand-written backward (PyTorch spec backend) vs autograd through the reference, float64."""
import pytest
import torch

from splat.camera import look_at
from splat.fast import render_fast
from splat.fast.binning import bin_gaussians as bin_fast
from splat.fast.preprocess import (PreprocessFn, preprocess_forward, quat_rotmat, rotmat_backward,
                                   sh_basis_jacobian)
from splat.gaussians import Gaussians
from splat.maths import covariance_3d
from splat.projection import project_gaussians
from splat.reference import bin_gaussians as bin_ref
from splat.reference import render
from splat.sh import rgb_to_sh0, sh_basis

pytestmark = pytest.mark.usefixtures("float64")
NAMES = ["means", "log_scales", "quats", "opacity_logits", "sh_dc", "sh_rest"]


def camera(w=40, h=30):
    return look_at((0.4, -3.0, 0.7), (0.0, 0.0, 0.0), width=w, height=h, fov_x_deg=55.0, dtype=torch.float64)


def scene(n=40, seed=0, opacity=(0.2, 0.95), extent=1.0, scales=(0.05, 0.3)):
    g = Gaussians.random(n, extent=extent, sh_degree=3, scale_range=scales, opacity_range=opacity, seed=seed,
                         dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed + 7)
    g.sh_dc[:, 0] = rgb_to_sh0(0.3 + 0.5 * torch.rand(n, 3, generator=gen, dtype=torch.float64))
    g.sh_rest *= 0.4
    return g


def test_sh_jacobian_matches_autograd():
    d = torch.nn.functional.normalize(torch.randn(20, 3, generator=torch.Generator().manual_seed(0)), dim=-1)
    J = sh_basis_jacobian(3, d)
    for i in range(20):
        Ja = torch.autograd.functional.jacobian(lambda v: sh_basis(3, v[None])[0], d[i])
        assert torch.allclose(J[i], Ja, atol=1e-12)


def test_rotmat_backward_matches_autograd():
    q = torch.nn.functional.normalize(torch.randn(10, 4, generator=torch.Generator().manual_seed(1)), dim=-1)
    gR = torch.randn(10, 3, 3, generator=torch.Generator().manual_seed(2))
    qq = q.clone().requires_grad_(True)
    (quat_rotmat(qq) * gR).sum().backward()
    assert torch.allclose(rotmat_backward(q, gR), qq.grad, atol=1e-12)


def test_preprocess_forward_matches_reference_projection():
    g = scene(seed=1)
    c = camera()
    (m2, cn, op, col, depth, radii, valid), _ = preprocess_forward(g.means, g.log_scales, g.quats,
                                                                   g.opacity_logits, g.sh, c, 3)
    p = project_gaussians(g.means, covariance_3d(g.log_scales, g.quats), c)
    assert torch.equal(valid, p.valid)
    assert torch.allclose(m2[valid], p.means2d[valid], atol=1e-12)
    assert torch.allclose(cn[valid], p.conics[valid], atol=1e-10)
    assert torch.equal(radii[valid], p.radii[valid])


@pytest.mark.parametrize("sh_degree", [0, 1, 3])
def test_preprocess_backward_matches_autograd(sh_degree):
    # wide extent: some Gaussians are behind the camera or outside the 1.3x FoV clamp
    g = scene(n=80, seed=2, extent=4.0)
    c = camera()
    _, sv = preprocess_forward(g.means, g.log_scales, g.quats, g.opacity_logits, g.sh, c, sh_degree)
    assert (sv.valid & (sv.tx_clamped | sv.ty_clamped)).any()   # FoV clamp branch exercised
    gen = torch.Generator().manual_seed(3)
    ups = None
    grads = []
    for hand in (True, False):
        ps = [p.clone().requires_grad_(True) for p in g.parameters().values()]
        if hand:
            outs = PreprocessFn.apply(*ps, c, sh_degree)
        else:
            outs, _ = preprocess_forward(ps[0], ps[1], ps[2], ps[3], torch.cat([ps[4], ps[5]], 1), c, sh_degree)
        m2, cn, op, col, _, _, valid = outs
        if ups is None:
            ups = [torch.randn(t.shape, generator=gen) for t in (m2, cn, op, col)]
        v = valid[:, None].to(m2.dtype)
        loss = ((m2 * ups[0] * v).sum() + (cn * ups[1] * v).sum() + (op * ups[2]).sum() + (col * ups[3]).sum())
        loss.backward()
        grads.append([p.grad for p in ps])
    assert (~valid).any() and valid.any()
    for name, a, b in zip(NAMES, *grads):
        assert torch.allclose(a, b, atol=1e-9, rtol=1e-7), (name, (a - b).abs().max())


def test_64bit_key_binning_matches_reference_binning():
    g = scene(n=80, seed=4, extent=1.5)
    c = camera(70, 45)
    (m2, cn, op, col, depth, radii, valid), _ = preprocess_forward(g.means, g.log_scales, g.quats,
                                                                   g.opacity_logits, g.sh, c, 3)
    fast = bin_fast(m2, radii, depth, valid, c.width, c.height)
    vis = torch.nonzero(valid).squeeze(-1)
    order = vis[torch.argsort(depth[vis], stable=True)]
    ref = bin_ref(m2[order], radii[order], c.width, c.height)
    assert torch.equal(fast.ranges, ref.ranges)
    assert torch.equal(fast.gauss_ids, order[ref.ranks])


def ref_and_fast_grads(g, c, bg, backend="torch", alpha_weight=0.0, **kw):
    gen = torch.Generator().manual_seed(9)
    up = torch.randn(c.height, c.width, 3, generator=gen)
    upa = torch.randn(c.height, c.width, generator=gen)
    res = []
    for fast in (False, True):
        ps = {k: v.clone().requires_grad_(True) for k, v in g.parameters().items()}
        gg = Gaussians(**ps)
        out = render_fast(gg, c, background=bg, backend=backend, **kw) if fast else \
            render(gg, c, background=bg, mode="tiled")
        ((out.image * up).sum() + alpha_weight * (out.alpha * upa).sum()).backward()
        res.append((out.image.detach(), out.alpha.detach(), {k: p.grad for k, p in ps.items()}))
    return res


@pytest.mark.parametrize("seed,opacity", [(5, (0.2, 0.9)), (6, (0.9, 0.99))])
def test_fast_torch_backend_matches_reference_forward_and_gradients(seed, opacity):
    g = scene(n=50, seed=seed, opacity=opacity)
    c = camera()
    bg = torch.tensor([0.3, 0.5, 0.8])
    (img_r, a_r, gr), (img_f, a_f, gf) = ref_and_fast_grads(g, c, bg, alpha_weight=0.7)
    assert torch.allclose(img_r, img_f, atol=1e-12)
    assert torch.allclose(a_r, a_f, atol=1e-12)
    for k in NAMES:
        assert torch.allclose(gr[k], gf[k], atol=1e-10, rtol=1e-7), (k, (gr[k] - gf[k]).abs().max())


def test_fast_torch_early_termination_and_batching():
    # many opaque, overlapping Gaussians: most pixels stop early
    g = scene(n=120, seed=8, opacity=(0.95, 0.99), scales=(0.2, 0.5), extent=0.5)
    c = camera(33, 20)
    bg = torch.tensor([1.0, 1.0, 1.0])
    (img_r, _, gr), (img_f, _, gf) = ref_and_fast_grads(g, c, bg, batch_elems=1)
    assert torch.allclose(img_r, img_f, atol=1e-12)
    for k in NAMES:
        assert torch.allclose(gr[k], gf[k], atol=1e-10, rtol=1e-7), k
