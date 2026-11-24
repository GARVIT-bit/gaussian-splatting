"""Rasterizer benchmark: my Triton rasterizer vs my PyTorch reference vs gsplat (reference only).

Measures, per (resolution, Gaussian count): forward time and forward+backward time of one
render (median over --reps after warm-up), peak GPU memory, and the image difference
between methods on the same scene. gsplat is imported only here, as a reference to compare
against; my renderer and trainer never import it.

    python bench/bench_rasterizer.py --out bench/results/phase3_rasterizer_random.md
    python bench/bench_rasterizer.py --scene runs/lego_800/final.pt --data <lego> --resolutions 800 \
        --out bench/results/phase3_rasterizer_lego.md
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from splat.camera import look_at  # noqa: E402
from splat.fast import render_fast  # noqa: E402
from splat.gaussians import Gaussians  # noqa: E402
from splat.reference import render as render_ref  # noqa: E402


def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return sha.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def random_scene(n, seed=0, device="cuda"):
    """Gaussians in a ball of radius 1 with small, varied scales (a stand-in for a trained scene)."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    r = torch.rand(n, 1, generator=g) ** (1 / 3)
    means = dirs * r
    log_scales = math.log(0.004) + torch.rand(n, 3, generator=g) * math.log(0.03 / 0.004)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    logits = torch.logit(0.1 + 0.8 * torch.rand(n, generator=g))
    sh_dc = torch.randn(n, 1, 3, generator=g) * 0.5
    sh_rest = torch.randn(n, 15, 3, generator=g) * 0.1
    return Gaussians(*(t.to(device).contiguous() for t in (means, log_scales, quats, logits, sh_dc, sh_rest)))


def timed(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), (min(ts), max(ts))


def make_gsplat():
    try:
        from gsplat import rasterization  # reference only
    except Exception as e:  # noqa: BLE001
        return None, f"not available ({type(e).__name__}: {e})"

    def run(g, cam, bg, sh_degree):
        w2c = torch.eye(4, device=g.means.device)
        w2c[:3, :3] = cam.R
        w2c[:3, 3] = cam.t
        K = torch.tensor([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], device=g.means.device,
                         dtype=torch.float32)
        img, alpha, _ = rasterization(g.means, g.quats, torch.exp(g.log_scales), torch.sigmoid(g.opacity_logits),
                                      g.sh, w2c[None], K[None], cam.width, cam.height, sh_degree=sh_degree,
                                      near_plane=0.2, eps2d=0.3, tile_size=16, backgrounds=bg[None],
                                      rasterize_mode="classic", packed=False)
        return img[0], alpha[0, ..., 0]
    try:
        import gsplat
        version = getattr(gsplat, "__version__", "?")
    except Exception:  # noqa: BLE001
        version = "?"
    return run, f"gsplat {version}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--counts", type=int, nargs="+", default=[10_000, 50_000, 100_000, 300_000])
    ap.add_argument("--scene", default=None, help="final.pt from scripts/train.py (overrides --counts)")
    ap.add_argument("--data", default=None, help="blender scene dir: use its first test camera")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ref-max-ms", type=float, default=4000, help="skip the reference where one render is slower")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "the benchmark needs a CUDA GPU"
    dev = "cuda"
    bg = torch.ones(3, device=dev)
    gs_run, gs_note = make_gsplat()

    if args.scene:
        params = torch.load(args.scene, map_location=dev)
        scenes = [("trained", Gaussians(**{k: v.to(dev).contiguous() for k, v in params.items()}))]
    else:
        scenes = [(f"random {n}", random_scene(n, device=dev)) for n in args.counts]

    rows = []
    for res in args.resolutions:
        if args.data:
            from splat.data import load_blender
            sc = load_blender(args.data, downscale=800 / res)
            cam = sc.test[0].camera
        else:
            cam = look_at((0.0, -3.0, 1.2), (0.0, 0.0, 0.0), width=res, height=res, fov_x_deg=50.0)
        for name, g in scenes:
            ps = {k: v.detach().clone().requires_grad_(True) for k, v in g.parameters().items()}
            gg = Gaussians(**ps)
            row = {"res": f"{cam.width}x{cam.height}", "scene": name, "n": g.num}

            def fw_t():
                with torch.no_grad():
                    return render_fast(gg, cam, sh_degree=args.sh_degree, background=bg, backend="triton")

            def fb_t():
                out = render_fast(gg, cam, sh_degree=args.sh_degree, background=bg, backend="triton")
                out.image.sum().backward()

            torch.cuda.reset_peak_memory_stats()
            row["triton_fwd"], _ = timed(fw_t, args.reps)
            row["triton_fb"], row["triton_fb_spread"] = timed(fb_t, args.reps)
            row["triton_mem"] = torch.cuda.max_memory_allocated() / 2**20
            img_t = fw_t().image
            row["pairs"] = int(fw_t().bins.gauss_ids.numel())

            # reference: one probe first, skip if too slow
            def fb_r():
                out = render_ref(gg, cam, sh_degree=args.sh_degree, background=bg, mode="tiled")
                out.image.sum().backward()
            t0 = time.perf_counter()
            fb_r()
            torch.cuda.synchronize()
            probe = (time.perf_counter() - t0) * 1000
            if probe < args.ref_max_ms:
                row["ref_fb"], _ = timed(fb_r, max(3, args.reps // 5), warmup=1)
                with torch.no_grad():
                    img_r = render_ref(gg, cam, sh_degree=args.sh_degree, background=bg, mode="tiled").image
                row["diff_ref"] = (img_r - img_t).abs().max().item()
            if gs_run is not None:
                def fw_g():
                    with torch.no_grad():
                        return gs_run(gg, cam, bg, args.sh_degree)

                def fb_g():
                    img, _ = gs_run(gg, cam, bg, args.sh_degree)
                    img.sum().backward()
                row["gs_fwd"], _ = timed(fw_g, args.reps)
                row["gs_fb"], row["gs_fb_spread"] = timed(fb_g, args.reps)
                row["diff_gs"] = (fw_g()[0] - img_t).abs().max().item()
            rows.append(row)
            print(row, flush=True)

    p = torch.cuda.get_device_properties(0)
    import triton
    f = lambda r, k, fmt="{:.1f}": fmt.format(r[k]) if k in r else "skipped"
    lines = [
        "# Phase 3: rasterizer benchmark (one render: forward, and forward + backward)",
        "",
        f"- Date: {datetime.date.today().isoformat()}  |  git commit: {git_commit()}",
        f"- GPU: {p.name}  |  PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  Triton {triton.__version__}"
        f"  |  Python {platform.python_version()}",
        f"- Reference comparison: {gs_note} (rasterize_mode classic, eps2d 0.3, near plane 0.2, tile 16)",
        f"- Scene: {'trained scene ' + args.scene if args.scene else 'random Gaussians in a unit ball, scales 0.004-0.03, opacity 0.1-0.9'}"
        f", SH degree {args.sh_degree}, white background",
        f"- Camera: {'first test camera of ' + os.path.basename(os.path.normpath(args.data)) if args.data else 'distance 3.2, FoV 50 deg'}",
        f"- Timing: median of {args.reps} (reference: {max(3, args.reps // 5)}) after warm-up, wall clock with "
        f"cuda.synchronize, includes preprocess + binning + sort. Loss for backward: image.sum().",
        "",
        "| resolution | scene | Gaussians | tile pairs | mine (Triton) fwd ms | mine fwd+bwd ms | reference fwd+bwd ms "
        "| gsplat fwd ms | gsplat fwd+bwd ms | speed-up vs reference | mine / gsplat (fwd+bwd) | peak MB (mine) "
        "| max abs diff mine vs ref | max abs diff mine vs gsplat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        sp = f"{r['ref_fb'] / r['triton_fb']:.1f}x" if "ref_fb" in r else "n/a"
        rel = f"{r['triton_fb'] / r['gs_fb']:.2f}x" if "gs_fb" in r else "n/a"
        lines.append(f"| {r['res']} | {r['scene']} | {r['n']} | {r['pairs']} | {f(r, 'triton_fwd')} | "
                     f"{f(r, 'triton_fb')} ({r['triton_fb_spread'][0]:.1f}-{r['triton_fb_spread'][1]:.1f}) | {f(r, 'ref_fb')} | {f(r, 'gs_fwd')} | {f(r, 'gs_fb') + (f" ({r['gs_fb_spread'][0]:.1f}-{r['gs_fb_spread'][1]:.1f})" if 'gs_fb' in r else '')} | {sp} | {rel} | "
                     f"{r['triton_mem']:.0f} | {f(r, 'diff_ref', '{:.1e}')} | {f(r, 'diff_gs', '{:.1e}')} |")
    lines += ["", "\"mine / gsplat\" above 1 means mine is slower. Reference rows are skipped where one render takes "
                  f"over {args.ref_max_ms / 1000:.0f} s. Forward+backward cells show median (min-max) over the reps."]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
