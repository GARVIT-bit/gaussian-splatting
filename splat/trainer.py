"""3DGS training: optimiser setup, LR schedule, adaptive density control, checkpoints.

Defaults follow the paper / official code (30k iterations):
  position LR 1.6e-4 -> 1.6e-6 (x scene extent), exponential decay over 30k steps
  SH DC LR 2.5e-3, higher SH 2.5e-3 / 20, opacity 0.025, scale 5e-3, rotation 1e-3
  loss (1 - 0.2) L1 + 0.2 D-SSIM, SH degree +1 every 1000 iterations up to 3
  densify every 100 its from 500 to 15k: view-space gradient threshold 2e-4,
  clone if max scale <= 1% of extent else split in 2 (scale / 1.6), prune
  opacity < 0.005; after the first opacity reset also prune Gaussians with a
  world scale > 10% of extent; opacity reset every 3000. The paper also prunes
  screen radius > 20 px, but in the official code that check sees radii that
  were just reset to 0 and never fires; we match the code (prune_by_screen_size).
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import Scene
from .gaussians import Gaussians
from .losses import gs_loss, psnr, ssim
from .reference import render as render_reference


def render(g, cam, *, sh_degree, background, mode, batch_elems):
    """Dispatch on TrainConfig.render_mode: "tiled" / "dense" (autograd reference),
    "fast_torch" (hand-written backward, PyTorch) or "triton" (hand-written backward, Triton)."""
    if mode in ("fast_torch", "triton"):
        from .fast import render_fast
        return render_fast(g, cam, sh_degree=sh_degree, background=background,
                           backend="torch" if mode == "fast_torch" else "triton", batch_elems=batch_elems)
    return render_reference(g, cam, sh_degree=sh_degree, background=background, mode=mode,
                            batch_elems=batch_elems)
from .sh import num_sh_coeffs, rgb_to_sh0

PARAM_NAMES = ("means", "log_scales", "quats", "opacity_logits", "sh_dc", "sh_rest")


@dataclass
class TrainConfig:
    iterations: int = 30_000
    sh_degree: int = 3
    sh_up_every: int = 1000
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_delay_steps: int = 0
    position_lr_max_steps: int = 30_000
    feature_lr: float = 2.5e-3
    opacity_lr: float = 0.025
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    lambda_dssim: float = 0.2
    percent_dense: float = 0.01
    densify_from: int = 500
    densify_until: int = 15_000
    densify_every: int = 100
    densify_grad_threshold: float = 2e-4
    opacity_reset_every: int = 3000
    min_opacity: float = 0.005
    max_screen_size: float = 20.0
    prune_by_screen_size: bool = False  # see DensityControl.densify_and_prune: matches the official code
    max_gaussians: int = 0          # 0 = no cap
    init_opacity: float = 0.1
    seed: int = 0
    render_mode: str = "tiled"      # tiled | dense | fast_torch | triton
    batch_elems: int = 1 << 22      # tiled reference: padded elements per tile batch
    device: str = "cpu"


def expon_lr(step: int, lr_init: float, lr_final: float, max_steps: int, delay_steps: int = 0,
             delay_mult: float = 1.0) -> float:
    """Log-linear interpolation from lr_init (step 0) to lr_final (max_steps), optional warm-up."""
    if step < 0 or (lr_init == 0 and lr_final == 0):
        return 0.0
    if delay_steps > 0:
        delay = delay_mult + (1 - delay_mult) * math.sin(0.5 * math.pi * min(max(step / delay_steps, 0), 1))
    else:
        delay = 1.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return delay * math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def knn_mean_sq_dist(x: torch.Tensor, k: int = 3, chunk: int = 1024) -> torch.Tensor:
    """Mean squared distance to the k nearest other points (for the initial scales)."""
    out = []
    for s in range(0, x.shape[0], chunk):
        d = torch.cdist(x[s:s + chunk], x) ** 2
        idx = torch.arange(s, min(s + chunk, x.shape[0]), device=x.device)
        d[torch.arange(len(idx), device=x.device), idx] = float("inf")  # exclude self
        out.append(d.topk(min(k, x.shape[0] - 1), largest=False).values.mean(-1))
    return torch.cat(out)


def init_from_points(points: np.ndarray, colors: np.ndarray, sh_degree: int, init_opacity: float = 0.1,
                     device="cpu", dtype=torch.float32) -> Gaussians:
    """Gaussians at the given points: isotropic scale = RMS distance to the 3 nearest neighbours,
    identity rotation, opacity 0.1, DC colour from the point colour, higher SH = 0."""
    xyz = torch.tensor(points, dtype=dtype, device=device)
    n = xyz.shape[0]
    dist2 = knn_mean_sq_dist(xyz).clamp_min(1e-7)
    log_scales = torch.log(torch.sqrt(dist2))[:, None].repeat(1, 3)
    quats = torch.zeros(n, 4, dtype=dtype, device=device)
    quats[:, 0] = 1
    opacity_logits = inverse_sigmoid(torch.full((n,), init_opacity, dtype=dtype, device=device))
    sh_dc = rgb_to_sh0(torch.tensor(colors, dtype=dtype, device=device))[:, None, :]
    sh_rest = torch.zeros(n, num_sh_coeffs(sh_degree) - 1, 3, dtype=dtype, device=device)
    return Gaussians(xyz, log_scales, quats, opacity_logits, sh_dc.contiguous(), sh_rest)


# ------------------------------------------------------------------ optimiser

class GaussianOptimizer:
    """Adam with one parameter group per Gaussian attribute, plus the tensor surgery
    density control needs: append new Gaussians (zero Adam moments), keep a subset
    (moments kept for survivors), and replace a tensor (moments reset)."""

    def __init__(self, g: Gaussians, cfg: TrainConfig, extent: float):
        self.g = g
        self.cfg = cfg
        self.extent = extent
        lrs = {"means": cfg.position_lr_init * extent, "log_scales": cfg.scaling_lr, "quats": cfg.rotation_lr,
               "opacity_logits": cfg.opacity_lr, "sh_dc": cfg.feature_lr, "sh_rest": cfg.feature_lr / 20}
        for name in PARAM_NAMES:
            getattr(g, name).requires_grad_(True)
        self.adam = torch.optim.Adam([{"params": [getattr(g, n)], "lr": lrs[n], "name": n} for n in PARAM_NAMES],
                                     lr=0.0, eps=1e-15)

    def update_lr(self, step: int):
        c = self.cfg
        lr = expon_lr(step, c.position_lr_init * self.extent, c.position_lr_final * self.extent,
                      c.position_lr_max_steps, c.position_lr_delay_steps, c.position_lr_delay_mult)
        for group in self.adam.param_groups:
            if group["name"] == "means":
                group["lr"] = lr
        return lr

    def _swap(self, group, new_param, state_fn):
        old = group["params"][0]
        st = self.adam.state.pop(old, None)
        new_param = new_param.detach().requires_grad_(True)
        if st is not None:
            st = {k: (state_fn(v) if torch.is_tensor(v) and v.shape == old.shape else v) for k, v in st.items()}
            self.adam.state[new_param] = st
        group["params"][0] = new_param
        setattr(self.g, group["name"], new_param)

    def append(self, new: dict[str, torch.Tensor]):
        for group in self.adam.param_groups:
            name = group["name"]
            ext = new[name].detach()
            self._swap(group, torch.cat([group["params"][0].detach(), ext]),
                       lambda v, e=ext: torch.cat([v, torch.zeros_like(e)]))

    def keep(self, mask: torch.Tensor):
        for group in self.adam.param_groups:
            self._swap(group, group["params"][0].detach()[mask], lambda v: v[mask])

    def replace(self, name: str, value: torch.Tensor):
        for group in self.adam.param_groups:
            if group["name"] == name:
                self._swap(group, value, torch.zeros_like)

    def step(self):
        self.adam.step()
        self.adam.zero_grad(set_to_none=True)


# ------------------------------------------------------------- density control

class DensityControl:
    """Accumulates the view-space positional gradient and runs clone / split / prune."""

    def __init__(self, n: int, device):
        self.reset(n, device)

    def reset(self, n: int, device):
        self.grad_accum = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)
        self.max_radii = torch.zeros(n, device=device)

    def add_stats(self, means2d_grad: torch.Tensor, radii: torch.Tensor, width: int, height: int):
        """`means2d_grad` is dL/d(pixel position). The official code thresholds the gradient
        w.r.t. NDC coordinates, which is dL/dpixel * (W/2, H/2); we convert so 2e-4 means the same."""
        vis = radii > 0
        g = means2d_grad[vis] * torch.tensor([width / 2, height / 2], device=means2d_grad.device,
                                             dtype=means2d_grad.dtype)
        self.grad_accum[vis] += g.norm(dim=-1)
        self.denom[vis] += 1
        self.max_radii[vis] = torch.maximum(self.max_radii[vis], radii[vis].to(self.max_radii.dtype))

    @torch.no_grad()
    def densify_and_prune(self, opt: GaussianOptimizer, cfg: TrainConfig, extent: float,
                          max_screen_size: float | None, generator: torch.Generator) -> dict:
        g = opt.g
        n0 = g.num
        grads = self.grad_accum / self.denom
        grads[~torch.isfinite(grads)] = 0.0
        scales = torch.exp(g.log_scales)
        max_scale = scales.max(-1).values
        selected = grads >= cfg.densify_grad_threshold
        clone = selected & (max_scale <= cfg.percent_dense * extent)
        split = selected & (max_scale > cfg.percent_dense * extent)

        if cfg.max_gaussians:  # respect the cap: keep the candidates with the largest gradients
            budget = max(cfg.max_gaussians - n0, 0)  # clone adds 1, split adds net 1
            cand = torch.nonzero(clone | split).squeeze(-1)
            if cand.numel() > budget:
                top = cand[torch.argsort(grads[cand], descending=True)[:budget]]
                allowed = torch.zeros_like(clone)
                allowed[top] = True
                clone &= allowed
                split &= allowed

        # clone: exact copies (the gradient step then moves the two apart)
        new = {name: getattr(g, name).detach()[clone] for name in PARAM_NAMES}
        # split: 2 samples from the Gaussian itself, scale / (0.8 * 2)
        n_split = int(split.sum())
        if n_split:
            from .maths import quat_to_rotmat
            std = scales[split].repeat(2, 1)
            samples = torch.randn(std.shape, generator=generator, device="cpu").to(std.device, std.dtype) * std
            R = quat_to_rotmat(g.quats.detach()[split]).repeat(2, 1, 1)
            split_new = {
                "means": (R @ samples[..., None])[..., 0] + g.means.detach()[split].repeat(2, 1),
                "log_scales": torch.log(scales[split] / (0.8 * 2)).repeat(2, 1),
            }
            for name in ("quats", "opacity_logits", "sh_dc", "sh_rest"):
                t = getattr(g, name).detach()[split]
                split_new[name] = t.repeat(2, *([1] * (t.dim() - 1)))
            new = {k: torch.cat([new[k], split_new[k]]) for k in PARAM_NAMES}
        n_clone = int(clone.sum())
        opt.append(new)

        # prune: split parents, low opacity, (after the first reset) huge Gaussians
        dev = g.means.device
        parents = torch.cat([split, torch.zeros(n_clone + 2 * n_split, dtype=torch.bool, device=dev)])
        prune = parents | (g.opacities.detach() < cfg.min_opacity)
        n_low = int((g.opacities.detach() < cfg.min_opacity).sum())
        if max_screen_size:
            big_world = torch.exp(g.log_scales.detach()).max(-1).values > 0.1 * extent
            prune |= big_world
            if cfg.prune_by_screen_size:
                # The paper's rule. Off by default: in the official code max_radii2D is zeroed by
                # densification_postfix() before this check, so it never prunes anything there.
                max_radii = torch.cat([self.max_radii, torch.zeros(n_clone + 2 * n_split, device=dev)])
                prune |= max_radii > max_screen_size
        opt.keep(~prune)
        self.reset(g.num, dev)
        return {"before": n0, "cloned": n_clone, "split": n_split, "pruned": int(prune.sum()) - n_split,
                "low_opacity": n_low, "after": g.num}

    @torch.no_grad()
    def reset_opacity(self, opt: GaussianOptimizer):
        g = opt.g
        new = inverse_sigmoid(torch.clamp_max(g.opacities.detach(), 0.01))
        opt.replace("opacity_logits", new)


# --------------------------------------------------------------------- trainer

class Trainer:
    def __init__(self, scene: Scene, init: Gaussians, cfg: TrainConfig, render_fn=None):
        """`render_fn(gaussians, camera, sh_degree=, background=)` replaces the renderer chosen by
        cfg.render_mode. Only the benchmark scripts use it (to train with gsplat's rasterizer as a
        reference); it must return .image and .projected.means2d / .projected.radii."""
        self.scene = scene
        self.cfg = cfg
        self.render_fn = render_fn
        self.device = torch.device(cfg.device)
        self.background = scene.background.to(self.device)
        g = Gaussians(**{k: v.detach().to(self.device).clone() for k, v in init.parameters().items()})
        self.opt = GaussianOptimizer(g, cfg, scene.extent)
        self.density = DensityControl(g.num, self.device)
        self.iteration = 0
        self.sh_degree = 0
        self.rng = random.Random(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.stack: list[int] = []
        self.log: list[dict] = []
        self.train_time = 0.0

    @property
    def gaussians(self) -> Gaussians:
        return self.opt.g

    def _render(self, cam):
        if self.render_fn is not None:
            return self.render_fn(self.gaussians, cam, sh_degree=self.sh_degree, background=self.background)
        return render(self.gaussians, cam, sh_degree=self.sh_degree, background=self.background,
                      mode=self.cfg.render_mode, batch_elems=self.cfg.batch_elems)

    def step(self) -> dict:
        cfg = self.cfg
        self.iteration += 1
        it = self.iteration
        t0 = time.perf_counter()
        self.opt.update_lr(it)
        if it % cfg.sh_up_every == 0 and self.sh_degree < min(cfg.sh_degree, self.gaussians.max_sh_degree):
            self.sh_degree += 1
        if not self.stack:
            self.stack = list(range(len(self.scene.train)))
        frame = self.scene.train[self.stack.pop(self.rng.randint(0, len(self.stack) - 1))]
        gt = frame.image_float(self.device, self.gaussians.means.dtype)

        out = self._render(frame.camera)
        out.projected.means2d.retain_grad()
        loss = gs_loss(out.image, gt, cfg.lambda_dssim)
        loss.backward()

        info = {"iter": it, "loss": loss.item(), "n": self.gaussians.num, "sh": self.sh_degree}
        with torch.no_grad():
            if it < cfg.densify_until:
                self.density.add_stats(out.projected.means2d.grad, out.projected.radii,
                                       frame.camera.width, frame.camera.height)
                if it > cfg.densify_from and it % cfg.densify_every == 0:
                    size = cfg.max_screen_size if it > cfg.opacity_reset_every else None
                    info["densify"] = self.density.densify_and_prune(self.opt, cfg, self.scene.extent, size,
                                                                      self.gen)
                if it % cfg.opacity_reset_every == 0 or (self.background.min() > 0.99 and it == cfg.densify_from):
                    self.density.reset_opacity(self.opt)
                    info["opacity_reset"] = True
        self.opt.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.train_time += dt
        info["time"] = dt
        return info

    # ---------------------------------------------------------- checkpointing

    def state_dict(self) -> dict:
        g = self.gaussians
        return {
            "iteration": self.iteration, "sh_degree": self.sh_degree, "config": dataclasses.asdict(self.cfg),
            "params": {k: v.detach().cpu() for k, v in g.parameters().items()},
            "adam": self.opt.adam.state_dict(),
            "density": {"grad_accum": self.density.grad_accum.cpu(), "denom": self.density.denom.cpu(),
                        "max_radii": self.density.max_radii.cpu()},
            "rng": self.rng.getstate(), "gen": self.gen.get_state(), "stack": list(self.stack),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "log": self.log, "train_time": self.train_time,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)  # atomic: a session dying mid-save never corrupts the last checkpoint

    def load(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        g = Gaussians(**{k: v.to(self.device) for k, v in sd["params"].items()})
        self.opt = GaussianOptimizer(g, self.cfg, self.scene.extent)
        self.opt.adam.load_state_dict(sd["adam"])
        self.density = DensityControl(g.num, self.device)
        for k, v in sd["density"].items():
            setattr(self.density, k, v.to(self.device))
        self.iteration, self.sh_degree = sd["iteration"], sd["sh_degree"]
        self.rng.setstate(sd["rng"])
        self.gen.set_state(sd["gen"])
        self.stack = list(sd["stack"])
        torch.set_rng_state(sd["torch_rng"])
        if sd["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(sd["cuda_rng"])
        self.log, self.train_time = sd["log"], sd["train_time"]

    # -------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self, frames=None, lpips_fn=None) -> dict:
        frames = self.scene.test if frames is None else frames
        ps, ss, ls = [], [], []
        for fr in frames:
            img = self._render(fr.camera).image.clamp(0, 1)
            gt = fr.image_float(self.device, self.gaussians.means.dtype)
            ps.append(psnr(img, gt))
            ss.append(ssim(img, gt).item())
            if lpips_fn is not None:
                ls.append(lpips_fn(img, gt))
        res = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n_views": len(frames),
               "psnr_per_view": ps}
        if ls:
            res["lpips"] = float(np.mean(ls))
        return res
