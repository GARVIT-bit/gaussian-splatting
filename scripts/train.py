"""Train a Gaussian-splatting scene, resuming from the last checkpoint if there is one.

    python scripts/train.py --data data/nerf_synthetic/lego --format blender --downscale 4 \
        --out runs/lego_ref_200 --name lego_ref_200 --max-gaussians 100000

Writes into --out: ckpt.pt (every --ckpt-every iterations and at the end), log.csv,
final.pt (parameters only), test renders; and bench/results/<name>.md (or --results).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from splat.data import load_blender, load_colmap, random_init_points  # noqa: E402
from splat.reference import render  # noqa: E402
from splat.trainer import TrainConfig, Trainer, init_from_points  # noqa: E402


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "splat", "scripts"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return (sha.stdout.strip() or "unknown") + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def hardware() -> str:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} ({p.total_memory / 2**30:.0f} GB), CUDA {torch.version.cuda}"
    return f"CPU {platform.processor() or platform.machine()}, {torch.get_num_threads()} threads"


def make_lpips(device):
    try:
        import lpips  # metric only
    except ImportError:
        return None, "not computed (lpips package not installed)"
    net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    def fn(img, gt):
        to = lambda x: x.permute(2, 0, 1)[None] * 2 - 1
        return net(to(img), to(gt)).item()
    return fn, "LPIPS (VGG), lpips package"


def save_png(t: torch.Tensor, path: str):
    Image.fromarray((t.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--format", choices=["blender", "colmap"], required=True)
    ap.add_argument("--downscale", type=float, default=1.0, help="blender: image downscale factor")
    ap.add_argument("--images", default="images", help="colmap: image folder (images_4 for Mip-NeRF 360 outdoor)")
    ap.add_argument("--white-background", action="store_true", default=None)
    ap.add_argument("--init-points", type=int, default=100_000, help="blender: random init points")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--iterations", type=int, default=30_000)
    ap.add_argument("--max-gaussians", type=int, default=0)
    ap.add_argument("--render-mode", default="tiled")
    ap.add_argument("--batch-elems", type=int, default=1 << 22)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--time-limit-hours", type=float, default=0, help="save and exit after this much wall time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-renders", type=int, default=4)
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.format == "blender":
        scene = load_blender(args.data, downscale=args.downscale,
                             white_background=True if args.white_background is None else args.white_background)
        pts, cols = random_init_points(args.init_points, seed=args.seed)
    else:
        scene = load_colmap(args.data, images_dir=args.images)
        pts, cols = scene.points, scene.colors
    c0 = scene.train[0].camera
    print(f"{len(scene.train)} train / {len(scene.test)} test views at {c0.width}x{c0.height}, "
          f"extent {scene.extent:.3f}, {len(pts)} init points", flush=True)

    sh_max = 3
    cfg = TrainConfig(iterations=args.iterations, max_gaussians=args.max_gaussians, seed=args.seed,
                      render_mode=args.render_mode, batch_elems=args.batch_elems, device=args.device,
                      sh_degree=sh_max, position_lr_max_steps=args.iterations,
                      densify_until=min(15_000, args.iterations // 2))
    init = init_from_points(pts, cols, sh_max, cfg.init_opacity, device=args.device)
    trainer = Trainer(scene, init, cfg)
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt):
        trainer.load(ckpt)
        print(f"resumed from {ckpt} at iteration {trainer.iteration}", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    log_path = os.path.join(args.out, "log.csv")
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as lf:
        w = csv.writer(lf)
        if new_log:
            w.writerow(["iter", "loss", "n_gaussians", "sh_degree", "sec_per_iter", "event"])
        window = []
        while trainer.iteration < cfg.iterations:
            info = trainer.step()
            window.append(info["time"])
            event = ""
            if "densify" in info:
                d = info["densify"]
                event = f"densify +{d['cloned']}c +{d['split']}s -{d['pruned']}p -> {d['after']}"
            if info.get("opacity_reset"):
                event += " opacity_reset"
            if info["iter"] % args.log_every == 0 or event:
                w.writerow([info["iter"], f"{info['loss']:.5f}", info["n"], info["sh"],
                            f"{np.mean(window):.4f}", event.strip()])
                lf.flush()
            if info["iter"] % args.log_every == 0:
                print(f"iter {info['iter']:6d} loss {info['loss']:.4f} n {info['n']:7d} sh {info['sh']} "
                      f"{np.mean(window):.3f} s/it {event}", flush=True)
                window = []
            if info["iter"] % args.ckpt_every == 0:
                trainer.save(ckpt)
            if args.time_limit_hours and time.time() - t_start > args.time_limit_hours * 3600:
                trainer.save(ckpt)
                print(f"time limit reached at iteration {trainer.iteration}; checkpoint saved, rerun to resume")
                return
    trainer.save(ckpt)
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else float("nan")

    # ---------------------------------------------------------------- evaluation
    lpips_fn, lpips_note = make_lpips(args.device)
    t_eval = time.time()
    res = trainer.evaluate(lpips_fn=lpips_fn)
    eval_time = time.time() - t_eval
    with torch.no_grad():
        n_r = min(args.n_renders, len(scene.test))
        t_r = time.perf_counter()
        for i in range(n_r):
            fr = scene.test[i]
            img = render(trainer.gaussians, fr.camera, sh_degree=trainer.sh_degree,
                         background=trainer.background, mode=cfg.render_mode, batch_elems=cfg.batch_elems).image
            save_png(torch.cat([img, fr.image_float(args.device)], 1), os.path.join(args.out, f"test_{i:03d}.png"))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        render_s = (time.perf_counter() - t_r) / max(n_r, 1)
    torch.save({k: v.detach().cpu() for k, v in trainer.gaussians.parameters().items()},
               os.path.join(args.out, "final.pt"))

    hours = trainer.train_time / 3600
    lines = [
        f"# {args.name}: training with the {cfg.render_mode} PyTorch reference renderer",
        "",
        f"- Date: {datetime.date.today().isoformat()}  |  git commit: {git_commit()}",
        f"- Hardware: {hardware()}  |  Python {platform.python_version()}  |  PyTorch {torch.__version__}",
        f"- Data: `{os.path.basename(os.path.normpath(args.data))}` ({args.format}), "
        f"{len(scene.train)} train / {len(scene.test)} test views at {c0.width}x{c0.height}"
        + (f" (downscale {args.downscale:g})" if args.format == "blender" else f" (`{args.images}`)")
        + f", background {scene.background.tolist()}",
        f"- Init: {len(pts)} {'random points in [-1.3, 1.3]^3' if args.format == 'blender' else 'COLMAP SfM points'}"
        f", seed {args.seed}, 1 run",
        f"- Iterations: {trainer.iteration}  |  Gaussian cap: {cfg.max_gaussians or 'none'}  |  final Gaussians: "
        f"{trainer.gaussians.num}  |  SH degree: {trainer.sh_degree}",
        f"- Training time (sum of iteration times, excl. loading/eval): {hours:.2f} h "
        f"({trainer.train_time / max(trainer.iteration, 1):.3f} s/iteration)",
        f"- Peak GPU memory allocated: {peak_mem:.2f} GB" if peak_mem == peak_mem else "- Peak GPU memory: n/a (CPU)",
        f"- Render time per test view (reference renderer, not optimised): {render_s:.3f} s",
        "",
        f"| metric | value |",
        f"|---|---:|",
        f"| test PSNR (dB) | {res['psnr']:.2f} |",
        f"| test SSIM | {res['ssim']:.4f} |",
        f"| test LPIPS | {res['lpips']:.4f} |" if "lpips" in res else f"| test LPIPS | {lpips_note} |",
        f"| test views | {res['n_views']} |",
        "",
        f"PSNR per view: min {min(res['psnr_per_view']):.2f}, max {max(res['psnr_per_view']):.2f}, "
        f"std {np.std(res['psnr_per_view']):.2f}. Evaluation took {eval_time:.0f} s. "
        f"{lpips_note if 'lpips' in res else ''}",
        "",
        "Full config:",
        "",
        "```json",
        json.dumps(dataclasses.asdict(cfg), indent=1),
        "```",
    ]
    out_md = args.results or os.path.join(ROOT, "bench", "results", f"{args.name}.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
