import pytest
import torch

from splat.camera import look_at
from splat.maths import covariance_3d
from splat.projection import LOW_PASS, perspective_jacobian, project_gaussians, world_to_camera

pytestmark = pytest.mark.usefixtures("float64")


def cam(width=64, height=48):
    return look_at((4.0, -3.0, 2.0), (0.0, 0.0, 0.0), width=width, height=height,
                   fov_x_deg=50.0, dtype=torch.float64)


def pixel_projection(t_cam, c):
    x, y, z = t_cam.unbind(-1)
    return torch.stack([c.fx * x / z + c.cx, c.fy * y / z + c.cy], -1)


def test_look_at_points_at_target():
    c = cam()
    p = world_to_camera(torch.zeros(1, 3), c)[0]
    assert torch.allclose(p[:2], torch.zeros(2), atol=1e-12)
    assert torch.isclose(p[2], torch.tensor(29.0).sqrt())
    uv = pixel_projection(p[None], c)[0]
    assert torch.allclose(uv, torch.tensor([c.cx, c.cy]))
    # world +z (up) appears towards the top of the image (smaller v)
    up = pixel_projection(world_to_camera(torch.tensor([[0.0, 0.0, 0.5]]), c), c)[0]
    assert up[1] < c.cy


def test_jacobian_matches_autograd():
    c = cam()
    g = torch.Generator().manual_seed(0)
    pts = torch.randn(20, 3, generator=g) * 0.5
    t_cam = world_to_camera(pts, c)
    J = perspective_jacobian(t_cam, c)
    for i in range(20):
        Ji = torch.autograd.functional.jacobian(lambda t: pixel_projection(t[None], c)[0], t_cam[i])
        assert torch.allclose(J[i], Ji, atol=1e-10)


def test_jacobian_matches_finite_differences():
    c = cam()
    t = world_to_camera(torch.tensor([[0.3, -0.2, 0.1]]), c)
    J = perspective_jacobian(t, c)[0]
    eps = 1e-6
    for k in range(3):
        e = torch.zeros(1, 3)
        e[0, k] = eps
        fd = (pixel_projection(t + e, c) - pixel_projection(t - e, c))[0] / (2 * eps)
        assert torch.allclose(J[:, k], fd, atol=1e-6)


def test_ewa_covariance_matches_monte_carlo_projection():
    """Sample a small 3D Gaussian, project the samples exactly, compare the 2D covariance."""
    c = cam(width=800, height=600)
    mean = torch.tensor([[0.2, 0.1, -0.1]])
    ls = torch.log(torch.tensor([[0.03, 0.01, 0.02]]))
    q = torch.tensor([[0.8, 0.3, -0.4, 0.2]])
    cov = covariance_3d(ls, q)
    proj = project_gaussians(mean, cov, c, low_pass=0.0)

    n = 400_000
    g = torch.Generator().manual_seed(1)
    L = torch.linalg.cholesky(cov[0])
    samples = mean + torch.randn(n, 3, generator=g) @ L.T
    uv = pixel_projection(world_to_camera(samples, c), c)
    emp_mean = uv.mean(0)
    emp_cov = torch.cov(uv.T)
    assert torch.allclose(emp_mean, proj.means2d[0], atol=0.05)
    rel = (emp_cov - proj.cov2d[0]).abs().max() / proj.cov2d[0].abs().max()
    assert rel < 0.02, rel


def test_low_pass_added_and_radius_covers_3_sigma():
    c = cam()
    mean = torch.tensor([[0.0, 0.0, 0.0]])
    cov = covariance_3d(torch.log(torch.tensor([[0.1, 0.05, 0.02]])), torch.tensor([[1.0, 0.2, 0.0, 0.1]]))
    p0 = project_gaussians(mean, cov, c, low_pass=0.0)
    p1 = project_gaussians(mean, cov, c)
    assert torch.allclose(p1.cov2d - p0.cov2d, LOW_PASS * torch.eye(2))
    lam = torch.linalg.eigvalsh(p1.cov2d[0]).max()
    assert p1.radii[0] >= 3 * lam.sqrt()
    assert p1.radii[0] <= 3 * lam.sqrt() + 1


def test_behind_camera_is_invalid_and_finite():
    c = cam()
    center = c.center
    forward = c.R[2]
    means = torch.stack([center - forward, center + 0.1 * forward, center + 3 * forward])
    cov = covariance_3d(torch.full((3, 3), -2.0), torch.tensor([[1.0, 0, 0, 0]] * 3))
    means.requires_grad_(True)
    p = project_gaussians(means, cov, c)
    assert p.valid.tolist() == [False, False, True]
    assert p.radii[:2].tolist() == [0, 0]
    (p.means2d.sum() + p.conics.sum()).backward()
    assert torch.isfinite(means.grad).all()
