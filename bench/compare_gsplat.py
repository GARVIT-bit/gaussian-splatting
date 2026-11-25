"""Diagnostic: why do my images and gsplat's differ on trained scenes (max 0.09) but not on random
ones (max 0.007)? Hypothesis: gsplat 1.5.3 clamps alpha at 0.999, mine (and 3DGS) at 0.99, which
only matters for nearly opaque Gaussians, and trained scenes have many.

Renders a trained scene from test views three ways: mine (alpha <= 0.99), mine with the clamp
moved to 0.999, and gsplat. If the hypothesis holds, mine@0.999 vs gsplat is much closer than
mine@0.99 vs gsplat. gsplat is imported only here, as a reference.

    python bench/compare_gsplat.py --scene runs/lego/final.pt --data <lego> --out bench/results/x.md
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from splat.data import load_blender, load_colmap  # noqa: E402
from splat.fast import render_fast  # noqa: E402
from splat.gaussians import Gaussians  # noqa: E402
from train_gsplat import gsplat_render  # noqa: E402


def stats(a, b):
    d = (a - b).abs().flatten().double()
    return {"max": d.max().item(), "mean": d.mean().item(), "p999": torch.quantile(d[:2_000_000], 0.999).item(),
            "frac_gt_1e-2": (d > 1e-2).double().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--format", choices=["blender", "colmap"], default="blender")
    ap.add_argument("--images", default="images")
    ap.add_argument("--views", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    params = torch.load(a.scene, map_location=dev)
    g = Gaussians(**{k: v.to(dev).contiguous() for k, v in params.items()})
    scene = load_blender(a.data) if a.format == "blender" else load_colmap(a.data, images_dir=a.images)
    bg = scene.background.to(dev)
    sh = g.max_sh_degree
    with torch.no_grad():
        op = torch.sigmoid(g.opacity_logits)
    pairs = {"mine@0.99 vs gsplat": [], "mine@0.999 vs gsplat": [], "mine@0.99 vs mine@0.999": []}
    with torch.no_grad():
        for fr in scene.test[:a.views]:
            m99 = render_fast(g, fr.camera, sh_degree=sh, background=bg, backend="triton").image
            m999 = render_fast(g, fr.camera, sh_degree=sh, background=bg, backend="triton", alpha_max=0.999).image
            gs = gsplat_render(g, fr.camera, sh_degree=sh, background=bg).image
            pairs["mine@0.99 vs gsplat"].append(stats(m99, gs))
            pairs["mine@0.999 vs gsplat"].append(stats(m999, gs))
            pairs["mine@0.99 vs mine@0.999"].append(stats(m99, m999))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    import gsplat
    lines = [
        "# Diagnostic: my rasterizer vs gsplat on a trained scene",
        "",
        f"- Date: {datetime.date.today().isoformat()}  |  git commit: {commit}  |  GPU: {torch.cuda.get_device_name(0)}"
        f"  |  gsplat {getattr(gsplat, '__version__', '?')}",
        f"- Scene: `{a.scene}` ({g.num} Gaussians; {100 * (op > 0.99).double().mean().item():.1f} % have opacity > 0.99),"
        f" {min(a.views, len(scene.test))} test views of `{os.path.basename(os.path.normpath(a.data))}`, "
        f"{scene.test[0].camera.width}x{scene.test[0].camera.height}",
        "",
        "| comparison | max abs diff (worst view) | mean abs diff | 99.9th percentile | pixels-channels > 0.01 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in pairs.items():
        lines.append(f"| {k} | {max(x['max'] for x in v):.2e} | {sum(x['mean'] for x in v) / len(v):.2e} | "
                     f"{sum(x['p999'] for x in v) / len(v):.2e} | {100 * sum(x['frac_gt_1e-2'] for x in v) / len(v):.3f} % |")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
