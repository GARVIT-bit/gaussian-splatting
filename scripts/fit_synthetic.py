"""Phase 1 "done" check: fit a small synthetic Gaussian scene on the CPU with the
autograd reference renderer, and report held-out PSNR / SSIM.

Writes bench/results/phase1_synthetic_fit.md and docs/images/phase1_synthetic_fit.png.

    .venv\\Scripts\\python.exe scripts/fit_synthetic.py
"""
from __future__ import annotations

import argparse
import datetime
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splat.camera import orbit_cameras  # noqa: E402
from splat.gaussians import Gaussians  # noqa: E402
from splat.reference import render  # noqa: E402
from splat.simple_fit import DEFAULT_LRS, fit_scene  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "splat", "scripts"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return (out.stdout.strip() or "no-commit") + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--gt-gaussians", type=int, default=40)
    ap.add_argument("--gaussians", type=int, default=200)
    ap.add_argument("--sh-degree", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "phase1_synthetic_fit.md"))
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    rows, histories = [], []
    for seed in args.seeds:
        gt = Gaussians.random(args.gt_gaussians, extent=0.7, sh_degree=args.sh_degree, scale_range=(0.05, 0.25),
                              opacity_range=(0.5, 0.95), seed=1000 + seed)
        cams = orbit_cameras(args.views, 3.0, width=args.res, height=args.res, fov_x_deg=50.0, seed=seed)
        # NeRF-style split: every 8th view held out
        test_idx = list(range(0, args.views, 8))
        train_idx = [i for i in range(args.views) if i not in test_idx]
        with torch.no_grad():
            imgs = [render(gt, c).image for c in cams]
        init = Gaussians.random(args.gaussians, extent=0.9, sh_degree=args.sh_degree, scale_range=(0.03, 0.08),
                                opacity_range=(0.1, 0.1), seed=seed)
        init.sh_dc.zero_()   # mid-grey
        init.sh_rest.zero_()
        t0 = time.perf_counter()
        res = fit_scene(init, [cams[i] for i in train_idx], [imgs[i] for i in train_idx],
                        [cams[i] for i in test_idx], [imgs[i] for i in test_idx],
                        iters=args.iters, seed=seed, eval_every=max(1, args.iters // 10),
                        log=lambda s: print(f"[seed {seed}] {s}", flush=True))
        dt = time.perf_counter() - t0
        rows.append((seed, res.init_test_psnr, res.test_psnr, res.test_ssim, dt))
        histories.append(res.history)
        if seed == args.seeds[0]:
            c = cams[test_idx[0]]
            with torch.no_grad():
                pred = render(res.model, c).image.clamp(0, 1)
                start = render(init, c).image.clamp(0, 1)
            strip = torch.cat([start, pred, imgs[test_idx[0]]], 1)
            strip = torch.nn.functional.interpolate(strip.permute(2, 0, 1)[None], scale_factor=4, mode="nearest")[0]
            os.makedirs(os.path.join(ROOT, "docs", "images"), exist_ok=True)
            Image.fromarray((strip.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
                os.path.join(ROOT, "docs", "images", "phase1_synthetic_fit.png"))

    finals = np.array([r[2] for r in rows])
    ssims = np.array([r[3] for r in rows])
    lines = [
        "# Phase 1: synthetic scene fit with the autograd reference renderer (CPU)",
        "",
        f"- Date: {datetime.date.today().isoformat()}  |  git commit: {git_commit()}",
        f"- CPU: {platform.processor() or platform.machine()}  |  threads: {torch.get_num_threads()}  |  "
        f"Python {platform.python_version()}  |  PyTorch {torch.__version__}",
        f"- Scene: {args.gt_gaussians} random ground-truth Gaussians (SH degree {args.sh_degree}) in [-0.7, 0.7]^3, "
        f"black background",
        f"- Cameras: {args.views} orbit views at radius 3, FoV 50 deg, {args.res}x{args.res} px; "
        f"every 8th view held out ({len(range(0, args.views, 8))} test views)",
        f"- Model: {args.gaussians} Gaussians, random init in [-0.9, 0.9]^3, grey, opacity 0.1; no densification "
        f"(Phase 2)",
        f"- Optimiser: Adam eps 1e-15, LRs {DEFAULT_LRS}, loss 0.8 L1 + 0.2 D-SSIM, {args.iters} iterations, "
        f"one view per iteration",
        f"- Seeds: {args.seeds} ({len(args.seeds)} runs)",
        "",
        "| seed | test PSNR at init (dB) | test PSNR final (dB) | test SSIM final | wall time (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {s} | {a:.2f} | {b:.2f} | {c:.4f} | {d:.0f} |" for s, a, b, c, d in rows]
    lines += ["", f"**Mean test PSNR {finals.mean():.2f} dB (min {finals.min():.2f}, max {finals.max():.2f}); "
                  f"mean SSIM {ssims.mean():.4f}** over {len(rows)} runs.", "",
              "Test PSNR during training (seed " + str(args.seeds[0]) + "):", "",
              "| iteration | train loss | test PSNR (dB) |", "|---:|---:|---:|"]
    lines += [f"| {it} | {l:.4f} | {p:.2f} |" for it, l, p in histories[0]]
    lines += ["", "Image: docs/images/phase1_synthetic_fit.png (init | fitted | ground truth, held-out view, "
                  f"seed {args.seeds[0]}, nearest-upsampled 4x)."]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
