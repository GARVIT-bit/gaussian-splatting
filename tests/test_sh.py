"""SH basis checked against the textbook real spherical harmonics formula.

Reference: Y_l^m(theta, phi) built from associated Legendre polynomials with the
Condon-Shortley phase,
    m > 0: sqrt(2) K_l^m cos(m phi)   P_l^m(cos theta)
    m = 0:         K_l^0              P_l^0(cos theta)
    m < 0: sqrt(2) K_l^|m| sin(|m| phi) P_l^|m|(cos theta)
    K_l^m = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!)
computed independently of the hard-coded polynomial constants in splat.sh.
"""
import math

import pytest
import torch

from splat.sh import eval_sh, num_sh_coeffs, rgb_to_sh0, sh_basis

pytestmark = pytest.mark.usefixtures("float64")


def assoc_legendre(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """P_l^m(x), m >= 0, with the Condon-Shortley phase, by the standard recurrence."""
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for _ in range(m):
            pmm = -pmm * fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pmmp1


def textbook_real_sh(l: int, m: int, dirs: torch.Tensor) -> torch.Tensor:
    x, y, z = dirs.unbind(-1)
    theta = torch.acos(z.clamp(-1, 1))
    phi = torch.atan2(y, x)
    am = abs(m)
    K = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - am) / math.factorial(l + am))
    P = assoc_legendre(l, am, torch.cos(theta))
    if m > 0:
        return math.sqrt(2) * K * torch.cos(m * phi) * P
    if m < 0:
        return math.sqrt(2) * K * torch.sin(am * phi) * P
    return K * P


def random_dirs(n, seed=0):
    d = torch.randn(n, 3, generator=torch.Generator().manual_seed(seed))
    return d / d.norm(dim=-1, keepdim=True)


def test_basis_matches_textbook_formula():
    dirs = random_dirs(500)
    basis = sh_basis(3, dirs)
    k = 0
    for l in range(4):
        for m in range(-l, l + 1):
            assert torch.allclose(basis[:, k], textbook_real_sh(l, m, dirs), atol=1e-12), (l, m)
            k += 1
    assert k == num_sh_coeffs(3) == 16


def test_basis_is_orthonormal_on_sphere():
    # Fibonacci-sphere quadrature: integral over S^2 ~ (4 pi / n) * sum
    n = 200_000
    i = torch.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = torch.sqrt(1 - z * z)
    phi = math.pi * (1 + math.sqrt(5)) * i
    dirs = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], -1)
    B = sh_basis(3, dirs)
    gram = (4 * math.pi / n) * B.T @ B
    assert torch.allclose(gram, torch.eye(16), atol=1e-4)


def test_lower_degree_is_prefix():
    dirs = random_dirs(50, seed=1)
    full = sh_basis(3, dirs)
    for deg in range(3):
        assert torch.allclose(sh_basis(deg, dirs), full[:, :num_sh_coeffs(deg)])


def test_eval_sh_dc_roundtrip_and_clamp():
    rgb = torch.rand(10, 3, generator=torch.Generator().manual_seed(2))
    coeffs = torch.zeros(10, 16, 3)
    coeffs[:, 0] = rgb_to_sh0(rgb)
    dirs = random_dirs(10, seed=3)
    for deg in range(4):
        assert torch.allclose(eval_sh(deg, coeffs, dirs), rgb)
    coeffs[:, 0] = -10.0
    assert (eval_sh(0, coeffs, dirs) == 0).all()


def test_eval_sh_ignores_inactive_coefficients():
    g = torch.Generator().manual_seed(4)
    coeffs = torch.randn(10, 16, 3, generator=g)
    dirs = random_dirs(10, seed=5)
    a = eval_sh(1, coeffs, dirs)
    coeffs[:, 4:] = 123.0
    assert torch.allclose(eval_sh(1, coeffs, dirs), a)
