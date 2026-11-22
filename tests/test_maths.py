import math

import pytest
import torch

from splat.maths import covariance_3d, inverse_2x2, quat_to_rotmat

pytestmark = pytest.mark.usefixtures("float64")


def test_identity_quaternion():
    R = quat_to_rotmat(torch.tensor([1.0, 0, 0, 0]))
    assert torch.allclose(R, torch.eye(3))


def test_90_degrees_about_z():
    h = math.sqrt(0.5)
    R = quat_to_rotmat(torch.tensor([h, 0, 0, h]))
    # rotating x onto y
    assert torch.allclose(R @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0]), atol=1e-12)
    assert torch.allclose(R @ torch.tensor([0.0, 0, 1]), torch.tensor([0.0, 0, 1]), atol=1e-12)


def test_rotation_orthonormal_and_norm_invariant():
    q = torch.randn(100, 4, generator=torch.Generator().manual_seed(1))
    R = quat_to_rotmat(q)
    I = torch.eye(3).expand(100, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), I, atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(100), atol=1e-12)
    assert torch.allclose(quat_to_rotmat(3.7 * q), R, atol=1e-12)
    assert torch.allclose(quat_to_rotmat(-q), R, atol=1e-12)  # q and -q are the same rotation


def test_axis_angle_matches_rodrigues():
    g = torch.Generator().manual_seed(2)
    axis = torch.randn(3, generator=g)
    axis = axis / axis.norm()
    angle = 1.234
    q = torch.cat([torch.tensor([math.cos(angle / 2)]), math.sin(angle / 2) * axis])
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_rod = torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    assert torch.allclose(quat_to_rotmat(q), R_rod, atol=1e-12)


def test_covariance_is_R_S2_Rt_and_psd():
    g = torch.Generator().manual_seed(3)
    ls = torch.randn(50, 3, generator=g)
    q = torch.randn(50, 4, generator=g)
    cov = covariance_3d(ls, q)
    R = quat_to_rotmat(q)
    S2 = torch.diag_embed(torch.exp(2 * ls))
    assert torch.allclose(cov, R @ S2 @ R.transpose(-1, -2), atol=1e-10)
    assert torch.allclose(cov, cov.transpose(-1, -2))
    eig = torch.linalg.eigvalsh(cov)
    assert (eig > 0).all()
    # eigenvalues are the squared scales
    assert torch.allclose(eig.sort(-1).values, torch.exp(2 * ls).sort(-1).values, rtol=1e-9)


def test_inverse_2x2():
    g = torch.Generator().manual_seed(4)
    A = torch.randn(20, 2, 2, generator=g)
    cov = A @ A.transpose(-1, -2) + 0.1 * torch.eye(2)
    conic, det = inverse_2x2(cov)
    inv = torch.linalg.inv(cov)
    assert torch.allclose(conic[:, 0], inv[:, 0, 0])
    assert torch.allclose(conic[:, 1], inv[:, 0, 1])
    assert torch.allclose(conic[:, 2], inv[:, 1, 1])
    assert torch.allclose(det, torch.linalg.det(cov))
