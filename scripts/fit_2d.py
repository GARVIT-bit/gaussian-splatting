"""Demo: fit the procedural test image with 2D Gaussians (same compositing as the 3D renderer).

Writes docs/images/phase1_fit2d.png (target | fit) and prints PSNR during training.

    .venv\\Scripts\\python.exe scripts/fit_2d.py
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splat.fit2d import fit_image, synthetic_pattern  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--gaussians", type=int, default=200)
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    target = synthetic_pattern(args.res, args.res)
    _, img, history = fit_image(target, n=args.gaussians, iters=args.iters, log_every=50)
    for it, p in history:
        print(f"iter {it:4d}  PSNR {p:.2f} dB")
    strip = torch.cat([target, img.clamp(0, 1)], 1)
    strip = torch.nn.functional.interpolate(strip.permute(2, 0, 1)[None], scale_factor=4, mode="nearest")[0]
    os.makedirs(os.path.join(ROOT, "docs", "images"), exist_ok=True)
    Image.fromarray((strip.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
        os.path.join(ROOT, "docs", "images", "phase1_fit2d.png"))


if __name__ == "__main__":
    main()
