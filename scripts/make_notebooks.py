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
sh(f"cd {REPO} && python -m pytest -p no:cacheprovider -rs tests")   # pytest.ini already adds -q
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


def phase3():
    cells = [
        md(r'''
# Phase 3: Triton rasterizer: GPU tests, benchmark, lego at full resolution

Runs on **Kaggle** with **GPU T4 x2** and Internet on. Run it with **Save Version → Save & Run All (Commit)**; every cell runs by itself.

1. Setup, then **all tests**, including the Triton kernels on the GPU: forward and every gradient vs the reference, in float64 (exact) and float32, plus a 30-iteration float64 training run that must match autograd. If any test fails, the notebook stops there.
2. Get lego. Try to install **gsplat**, used only by the benchmark script as a reference. If it can't be built here, the benchmark records "not available".
3. Two jobs in parallel, one per GPU:
   - **GPU 0:** lego at **800x800**, 30k iterations, paper settings, no Gaussian cap, my Triton rasterizer (Phase 3 "done"). Then the benchmark on that trained scene at 800x800.
   - **GPU 1:** lego at **200x200** with *exactly* the Phase-2 settings, but on the Triton rasterizer, to compare speed and quality with the Phase-2 reference run. Then the random-scene benchmark (200/400/800 px x 10k-300k Gaussians).
4. Package everything into `phase3_results.zip`.

**GPU-hour estimate (a guess until measured):** about 1.5-4 h of session time. Kaggle counts the session's wall clock once, even with both GPUs in use. Both training runs checkpoint every 1000 iterations and stop at 10.5 h. If the session ends early, see the resume cell.
'''),
        code(SETUP),
        code(r'''
# Unit tests (CPU) + GPU tests: Triton kernels vs the reference, CUDA consistency
sh(f"cd {REPO} && python -m pytest -p no:cacheprovider -rs tests")
'''),
        code(LEGO_DATA),
        code(r'''
# gsplat: reference for the benchmark only (never imported by my renderer or trainer).
# Building its CUDA code can take several minutes; failure is recorded, not fatal.
r = subprocess.run("pip install -q gsplat && python -c \"import torch, gsplat; from gsplat import rasterization; "
                   "m=torch.zeros(1,3,device='cuda'); q=torch.tensor([[1.,0,0,0]],device='cuda'); "
                   "s=torch.full((1,3),.1,device='cuda'); o=torch.ones(1,device='cuda'); c=torch.ones(1,3,device='cuda'); "
                   "V=torch.eye(4,device='cuda'); V[2,3]=3; K=torch.tensor([[50.,0,32],[0,50,32],[0,0,1]],device='cuda'); "
                   "print('gsplat', gsplat.__version__, rasterization(m,q,s,o,c,V[None],K[None],64,64)[0].shape)\"",
                   shell=True, timeout=3600)
print("gsplat ready" if r.returncode == 0 else "gsplat NOT available; the benchmark will record that")
'''),
        md(r'''
## Resume (only when continuing an interrupted run)

On Kaggle: add the earlier version's **output** as an input (Add Input → Your Work → this notebook → that version), then run all again. This cell copies the `runs/` folders (with `ckpt.pt`) back, and training continues from the checkpoints.
'''),
        code(r'''
RUNS = os.path.join(WORK, "runs")
os.makedirs(RUNS, exist_ok=True)
for name in ("lego_triton_800", "lego_triton_200"):
    prev = glob.glob(f"/kaggle/input/**/runs/{name}/ckpt.pt", recursive=True)
    dst = os.path.join(RUNS, name)
    if prev and not os.path.exists(os.path.join(dst, "ckpt.pt")):
        shutil.copytree(os.path.dirname(prev[0]), dst, dirs_exist_ok=True)
        print("resuming", name, "from", prev[0])
'''),
        md("## Two jobs in parallel (one per GPU)"),
        code(r'''
RES = os.path.join(WORK, "phase3_results")
os.makedirs(RES, exist_ok=True)
train = (f"python scripts/train.py --data {LEGO} --format blender --render-mode triton --iterations 30000 "
         f"--ckpt-every 1000 --log-every 500 --time-limit-hours 10.5")
jobs = {
    # GPU 0: full resolution (Phase 3 "done"), then the benchmark on the trained scene
    0: (f"{train} --downscale 1 --out {RUNS}/lego_triton_800 --name lego_triton_800 "
        f"--results {RES}/lego_triton_800.md && "
        f"python bench/bench_rasterizer.py --scene {RUNS}/lego_triton_800/final.pt --data {LEGO} "
        f"--resolutions 200 400 800 --out {RES}/phase3_rasterizer_lego.md"),
    # GPU 1: exactly the Phase-2 settings on the Triton rasterizer, then the random-scene benchmark
    1: (f"{train} --downscale 4 --max-gaussians 200000 --out {RUNS}/lego_triton_200 --name lego_triton_200 "
        f"--results {RES}/lego_triton_200.md && "
        f"python bench/bench_rasterizer.py --out {RES}/phase3_rasterizer_random.md"),
}
procs = {}
for gpu, cmd in jobs.items():
    log = open(os.path.join(RES, f"gpu{gpu}.log"), "w")
    print(f"GPU {gpu}: {cmd}\n")
    procs[gpu] = subprocess.Popen(f"cd {REPO} && {cmd}", shell=True, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
t0 = time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(300)
    tails = {g: open(os.path.join(RES, f"gpu{g}.log")).read().strip().splitlines()[-1:] for g in procs}
    print(f"[{(time.time() - t0) / 3600:.2f} h]", {g: t[0][:120] if t else "" for g, t in tails.items()}, flush=True)
for g, p in procs.items():
    print(f"GPU {g} exit code {p.returncode}")
    print("".join(open(os.path.join(RES, f"gpu{g}.log")).readlines()[-25:]))
'''),
        md("## Package results for download"),
        code(r'''
for name in ("lego_triton_800", "lego_triton_200"):
    d = os.path.join(RUNS, name)
    for f in glob.glob(os.path.join(d, "*")):
        if f.endswith(("log.csv", ".png")):
            shutil.copy(f, os.path.join(RES, f"{name}_{os.path.basename(f)}"))
    if os.path.exists(os.path.join(d, "final.pt")):
        shutil.copy(os.path.join(d, "final.pt"), os.path.join(RES, f"{name}_final.pt"))
shutil.make_archive(RES, "zip", RES)
print("download:", RES + ".zip")
for f in sorted(glob.glob(os.path.join(RES, "*.md"))):
    print("=" * 80, "\n", open(f).read())
'''),
    ]
    return notebook(cells)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    for name, fn in (("phase2_lego", phase2), ("phase3_triton", phase3)):
        path = os.path.join(ROOT, "notebooks", f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fn(), f, indent=1)
        print("wrote", path)
