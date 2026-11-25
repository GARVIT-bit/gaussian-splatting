"""Reference run: MY training loop (same data, density control, optimiser, schedule, seed and
evaluation as scripts/train.py) with gsplat's rasterizer in place of mine.

Only this benchmark script imports gsplat. The comparison then isolates the rasterizer: anything
that differs between `train.py --render-mode triton` and this script on the same scene is gsplat's
rasterizer (speed, and its small numeric differences such as the 0.999 alpha clamp).

    python bench/train_gsplat.py --data <lego> --format blender --out runs/lego_gsplat --name lego_gsplat ...
(all arguments are the same as scripts/train.py; --render-mode is ignored)
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import train  # noqa: E402  (scripts/train.py)


class _GradView:
    """Exposes gsplat's (1, N, 2) means2d as the (N, 2) tensor the trainer's density stats expect."""

    def __init__(self, m):
        self._m = m

    def retain_grad(self):
        self._m.retain_grad()

    @property
    def grad(self):
        return None if self._m.grad is None else self._m.grad[0]


class _Projected:
    def __init__(self, means2d, radii):
        self.means2d = _GradView(means2d)
        self.radii = radii


class _Out:
    def __init__(self, image, alpha, projected):
        self.image, self.alpha, self.projected = image, alpha, projected


def gsplat_render(g, cam, *, sh_degree, background):
    from gsplat import rasterization  # reference only

    dev = g.means.device
    w2c = torch.eye(4, device=dev)
    w2c[:3, :3] = cam.R.to(dev)
    w2c[:3, 3] = cam.t.to(dev)
    K = torch.tensor([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], device=dev, dtype=torch.float32)
    img, alpha, meta = rasterization(
        g.means, g.quats, torch.exp(g.log_scales), torch.sigmoid(g.opacity_logits), g.sh,
        w2c[None], K[None], cam.width, cam.height, sh_degree=sh_degree, near_plane=0.2, eps2d=0.3,
        tile_size=16, backgrounds=background[None], rasterize_mode="classic", packed=False, absgrad=False)
    radii = meta["radii"][0]
    if radii.dim() == 2:  # gsplat >= 1.5: per-axis radii (N, 2); visible iff both > 0
        radii = torch.where((radii > 0).all(-1), radii.max(-1).values, torch.zeros_like(radii[:, 0]))
    return _Out(img[0], alpha[0, ..., 0], _Projected(meta["means2d"], radii.long()))


def main():
    import gsplat
    label = f"gsplat {getattr(gsplat, '__version__', '?')} rasterization (reference), my training loop"
    argv = [a for a in sys.argv[1:]]
    train.main(argv, render_fn=gsplat_render, renderer_label=label)


if __name__ == "__main__":
    main()
