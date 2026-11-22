import torch

from splat.losses import gs_loss, psnr, ssim


def test_ssim_identity_and_ordering():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(32, 32, 3, generator=g, dtype=torch.float64)
    assert torch.isclose(ssim(x, x), torch.tensor(1.0, dtype=torch.float64))
    small = (x + 0.02 * torch.randn(x.shape, generator=g, dtype=torch.float64)).clamp(0, 1)
    big = (x + 0.2 * torch.randn(x.shape, generator=g, dtype=torch.float64)).clamp(0, 1)
    assert 1 > ssim(x, small) > ssim(x, big)
    assert torch.isclose(ssim(x, big), ssim(big, x))


def test_loss_and_psnr():
    x = torch.full((8, 8, 3), 0.5)
    assert gs_loss(x, x).abs() < 1e-6
    assert abs(psnr(x, x + 0.1) - 20.0) < 1e-4
