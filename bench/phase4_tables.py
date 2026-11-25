"""Aggregate Phase-4 runs into results tables: mine vs gsplat (same training loop, same T4, measured)
vs the 3DGS paper (reported, different hardware), per dataset.

    python bench/phase4_tables.py --runs bench/results/phase4 --out bench/results/phase4_summary.md

Reads every *.json written by scripts/train.py / bench/train_gsplat.py in --runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BLENDER = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
MIP360 = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
TANDT = ["truck", "train"]


def dataset_of(scene: str) -> str:
    return "nerf_synthetic" if scene in BLENDER else "mipnerf360" if scene in MIP360 else \
        "tanks_temples" if scene in TANDT else "own"


def who(run: dict) -> str:
    return "gsplat" if run["renderer"].startswith("gsplat") else "mine" if run["renderer"] == "triton" else run["renderer"]


def fmt(x, nd):
    return "–" if x is None else f"{x:.{nd}f}"


def build(runs: list[dict], paper: dict) -> list[str]:
    table = defaultdict(dict)                       # (dataset, scene) -> who -> run
    for r in runs:
        table[(dataset_of(r["data"]), r["data"])][who(r)] = r
    lines = ["# Phase 4: results", "",
             "Measured columns: my code on a Kaggle Tesla T4, 30k iterations, one run (seed 0) per scene, settings in "
             "each run's results file. \"gsplat\" = the same training loop with gsplat's rasterizer (bench/train_gsplat.py), "
             "same T4. **Paper = reported in Kerbl et al. 2023 (A6000, their code); not measured here** "
             "(bench/paper_numbers.json says how each paper value was read).", ""]
    for ds, title in (("nerf_synthetic", "NeRF-synthetic (800x800, white background, test split)"),
                      ("mipnerf360", "Mip-NeRF 360 (images_4 outdoor / images_2 indoor, every 8th image held out)"),
                      ("tanks_temples", "Tanks & Temples (3DGS release images, every 8th image held out)"),
                      ("own", "Own capture")):
        scenes = sorted(s for (d, s) in table if d == ds)
        if not scenes:
            continue
        p = paper.get(ds, {})
        lines += [f"## {title}", "",
                  "| scene | resolution | mine PSNR | mine SSIM | mine LPIPS | gsplat PSNR | gsplat SSIM | gsplat LPIPS "
                  "| paper PSNR (reported) | paper SSIM (reported) | paper LPIPS (reported) | mine Gaussians "
                  "| mine train h | gsplat train h | mine render ms |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        acc = defaultdict(list)
        for s in scenes:
            m, g = table[(ds, s)].get("mine"), table[(ds, s)].get("gsplat")
            ref = m or g
            pp, ps, pl = (p.get(k, {}).get(s) for k in ("psnr", "ssim", "lpips"))
            lines.append(
                f"| {s} | {ref['width']}x{ref['height']} | {fmt(m and m['psnr'], 2)} | {fmt(m and m['ssim'], 3)} | "
                f"{fmt(m and m['lpips'], 3)} | {fmt(g and g['psnr'], 2)} | {fmt(g and g['ssim'], 3)} | "
                f"{fmt(g and g['lpips'], 3)} | {fmt(pp, 2)} | {fmt(ps, 3)} | {fmt(pl, 3)} | "
                f"{m['gaussians'] if m else '–'} | {fmt(m and m['train_hours'], 2)} | {fmt(g and g['train_hours'], 2)} | "
                f"{fmt(m and m['render_ms'], 1)} |")
            for key, run in (("mine", m), ("gsplat", g)):
                if run:
                    for k in ("psnr", "ssim", "lpips", "train_hours"):
                        if run.get(k) is not None:
                            acc[(key, k)].append(run[k])
            for k, v in (("psnr", pp), ("ssim", ps), ("lpips", pl)):
                if v is not None:
                    acc[("paper", k)].append(v)
        mean = lambda key, k, nd: fmt(sum(acc[(key, k)]) / len(acc[(key, k)]), nd) if acc[(key, k)] else "–"
        n = lambda key: len(acc[(key, "psnr")])
        lines += [f"| **mean of the scenes above** | | {mean('mine', 'psnr', 2)} | {mean('mine', 'ssim', 3)} | "
                  f"{mean('mine', 'lpips', 3)} | {mean('gsplat', 'psnr', 2)} | {mean('gsplat', 'ssim', 3)} | "
                  f"{mean('gsplat', 'lpips', 3)} | {mean('paper', 'psnr', 2)} | {mean('paper', 'ssim', 3)} | "
                  f"{mean('paper', 'lpips', 3)} | | {mean('mine', 'train_hours', 2)} | {mean('gsplat', 'train_hours', 2)} | |",
                  "", f"Scenes with results: mine {n('mine')}, gsplat {n('gsplat')}, paper values {n('paper')}. "
                      "Means only over the scenes listed, so they are comparable only when the counts match.", ""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "bench", "results", "phase4"))
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "phase4_summary.md"))
    a = ap.parse_args()
    runs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(a.runs, "*.json")))]
    paper = json.load(open(os.path.join(ROOT, "bench", "paper_numbers.json"), encoding="utf-8"))
    lines = build(runs, paper)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
