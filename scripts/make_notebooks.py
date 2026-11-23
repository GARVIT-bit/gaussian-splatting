"""Generate the Kaggle/Colab notebooks in notebooks/ (keeps them reviewable as plain Python).

    .venv\\Scripts\\python.exe scripts/make_notebooks.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"},
                                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = r'''
import os, sys, subprocess, shutil, glob, time
KAGGLE = os.path.exists("/kaggle/working")
COLAB = "google.colab" in sys.modules
WORK = "/kaggle/working" if KAGGLE else ("/content" if COLAB else os.path.abspath("."))
DATA = "/tmp/data" if KAGGLE else os.path.join(WORK, "data")   # outside /kaggle/working: not saved as output
REPO_URL = "https://github.com/GARVIT-bit/gaussian-splatting.git"
BRANCH = "main"
REPO = os.path.join(WORK, "gaussian-splatting")
print("platform:", "kaggle" if KAGGLE else "colab" if COLAB else "local", "| work dir:", WORK)

def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")

if not os.path.exists(REPO):
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
else:
    sh(f"git -C {REPO} pull --ff-only")
sh(f"git -C {REPO} log -1 --format='%h %s'")
sh("pip install -q lpips pytest")   # torch, numpy, pillow are preinstalled on Kaggle and Colab
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
os.makedirs(DATA, exist_ok=True)
'''

TESTS = r'''
# All CPU unit tests + the CUDA consistency test (tiled reference, float32 GPU vs float64 CPU)
sh(f"cd {REPO} && python -m pytest -q -p no:cacheprovider tests")
'''

LEGO_DATA = r'''
# NeRF-synthetic lego. If you attached a Kaggle dataset that contains nerf_synthetic/lego, it is used;
# otherwise lego is extracted from the NeRF authors' example zip (370 MB download, lego only).
found = glob.glob("/kaggle/input/**/nerf_synthetic/lego/transforms_train.json", recursive=True)
if found:
    LEGO = os.path.dirname(found[0])
else:
    LEGO = os.path.join(DATA, "nerf_synthetic", "lego")
    if not os.path.exists(os.path.join(LEGO, "transforms_train.json")):
        zp = os.path.join(DATA, "nerf_example_data.zip")
        sh(f"wget -q -O {zp} http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip")
        sh(f"cd {DATA} && unzip -q -o {zp} && rm {zp}")
        LEGO = os.path.dirname(glob.glob(os.path.join(DATA, "**", "lego", "transforms_train.json"), recursive=True)[0])
print("lego at", LEGO, "|", len(os.listdir(os.path.join(LEGO, "train"))), "train images")
'''


def phase2():
    cells = [
        md(r'''
# Phase 2: train NeRF-synthetic lego with the PyTorch reference renderer

Runs on **Kaggle** (GPU T4, Internet on) or **Colab** (T4 runtime).

1. Setup: clone the repo, install `lpips` and `pytest`.
2. Run all unit tests, including the CUDA check.
3. Get lego.
4. A **300-iteration timing run** that prints seconds/iteration and the projected time for the full run.
5. **The full run:** lego at 200x200 (4x downscale), 30k iterations, paper hyperparameters, 100k random init points, at most 200k Gaussians. It checkpoints every 1000 iterations and stops cleanly before the 12 h session limit.
6. Package the results to download.

**GPU-hour estimate:** setup + tests + timing ~0.3 h. The full run is a guess of 3-8 h (0.3-1 s/it); cell 4 prints the real projection. If the projection is above ~11 h, the run stops at the time limit with a checkpoint, and running the notebook again resumes it (see the resume cell).
'''),
        code(SETUP),
        code(TESTS),
        code(LEGO_DATA),
        md("## Timing run (300 iterations, not reported)"),
        code(r'''
RUNS = os.path.join(WORK, "runs")
dbg = os.path.join(RUNS, "lego_ref_200_timing")
shutil.rmtree(dbg, ignore_errors=True)
sh(f"cd {REPO} && python scripts/train.py --data {LEGO} --format blender --downscale 4 --out {dbg} "
   f"--name timing --results {dbg}/timing.md --iterations 300 --max-gaussians 200000 --log-every 50 --n-renders 1")
import csv
rows = list(csv.DictReader(open(os.path.join(dbg, "log.csv"))))
spi = float(rows[-1]["sec_per_iter"])
print(f"\n{spi:.3f} s/iteration (100k Gaussians, before any densification)"
      f" -> 30k iterations ~ {spi * 30000 / 3600:.1f} h at this Gaussian count (densification will change it)")
'''),
        md(r'''
## Resume (only when continuing an earlier, interrupted run)

On Kaggle: add the earlier notebook version's **output** as an input (Add Input → Your Work → this notebook → that version), then run this cell. It copies `runs/lego_ref_200` (with `ckpt.pt`) back into the working dir and the next cell continues from the checkpoint. On Colab, upload the folder to `/content/runs/lego_ref_200` instead.
'''),
        code(r'''
prev = glob.glob("/kaggle/input/**/runs/lego_ref_200/ckpt.pt", recursive=True)
dst = os.path.join(RUNS, "lego_ref_200")
if prev and not os.path.exists(os.path.join(dst, "ckpt.pt")):
    shutil.copytree(os.path.dirname(prev[0]), dst, dirs_exist_ok=True)
    print("resuming from", prev[0])
else:
    print("no earlier checkpoint found; starting fresh" if not prev else "checkpoint already in place")
'''),
        md("## Full run (reported in bench/results/lego_ref_200.md)"),
        code(r'''
out = os.path.join(RUNS, "lego_ref_200")
LIMIT_H = 11.0 if KAGGLE else 0   # Kaggle sessions end at 12 h; Colab: rely on checkpoints
sh(f"cd {REPO} && python scripts/train.py --data {LEGO} --format blender --downscale 4 --out {out} "
   f"--name lego_ref_200 --results {out}/lego_ref_200.md --iterations 30000 --max-gaussians 200000 "
   f"--ckpt-every 1000 --log-every 500 --time-limit-hours {LIMIT_H}")
'''),
        md("## Package results for download"),
        code(r'''
pkg = os.path.join(WORK, "phase2_results")
os.makedirs(pkg, exist_ok=True)
for f in glob.glob(os.path.join(RUNS, "lego_ref_200", "*")):
    if not f.endswith(("ckpt.pt", ".tmp")):
        shutil.copy(f, pkg)
shutil.make_archive(pkg, "zip", pkg)
print("download:", pkg + ".zip")
print(open(os.path.join(pkg, "lego_ref_200.md")).read() if os.path.exists(os.path.join(pkg, "lego_ref_200.md"))
      else "run not finished yet: run the notebook again to resume")
if COLAB:
    from google.colab import files
    files.download(pkg + ".zip")
'''),
    ]
    return notebook(cells)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    path = os.path.join(ROOT, "notebooks", "phase2_lego.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(phase2(), f, indent=1)
    print("wrote", path)
