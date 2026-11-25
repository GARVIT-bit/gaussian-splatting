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

Runs on **Kaggle** with **GPU T4 x2** and Internet on: use **Save Version → Save & Run All (Commit)**, and every cell runs by itself. It also runs on **Colab** (Runtime → Change runtime type → T4 GPU → Run all). With one GPU the two jobs below run one after the other instead of in parallel; keep the Colab tab open.

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
        md("## Two jobs: in parallel with 2 GPUs (Kaggle), one after the other with 1 GPU (Colab)"),
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
import torch
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
if n_gpu < 2:   # e.g. Colab's single T4: same two jobs, one after the other (full resolution first)
    jobs = {0: f"({jobs[0]}) ; ({jobs[1]})"}
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


GSPLAT_INSTALL = r'''
# gsplat: reference only (imported by bench/train_gsplat.py and bench/compare_gsplat.py, never by my code).
r = subprocess.run("pip install -q gsplat && python -c \"import torch, gsplat; from gsplat import rasterization; "
                   "m=torch.zeros(1,3,device='cuda'); q=torch.tensor([[1.,0,0,0]],device='cuda'); "
                   "s=torch.full((1,3),.1,device='cuda'); o=torch.ones(1,device='cuda'); c=torch.ones(1,3,device='cuda'); "
                   "V=torch.eye(4,device='cuda'); V[2,3]=3; K=torch.tensor([[50.,0,32],[0,50,32],[0,0,1]],device='cuda'); "
                   "print('gsplat', gsplat.__version__, rasterization(m,q,s,o,c,V[None],K[None],64,64)[0].shape)\"",
                   shell=True, timeout=3600)
if r.returncode != 0:
    raise RuntimeError("gsplat could not be installed/built in this session: the gsplat runs cannot happen")
'''

PHASE4_RUN = r'''
RUNS = os.path.join(WORK, "runs")
RES = os.path.join(WORK, f"phase4_{SESSION}_results")
os.makedirs(RUNS, exist_ok=True)
os.makedirs(RES, exist_ok=True)

def job_cmd(j):
    ds, scene = j["dataset"], j["scene"]
    if ds == "blender":
        data = f"--data {DATA}/nerf_synthetic/{scene} --format blender --downscale 1"
    elif ds == "mipnerf360":
        folder = "images_4" if scene in ("bicycle", "flowers", "garden", "stump", "treehill") else "images_2"
        data = f"--data {DATA}/360_v2/{scene} --format colmap --images {folder} --max-gaussians {CAP}"
    else:
        data = f"--data {DATA}/tandt/{scene} --format colmap --images images --max-gaussians {CAP}"
    script = "scripts/train.py --render-mode triton" if j["who"] == "mine" else "bench/train_gsplat.py"
    name = j["name"]
    return (f"python {script} {data} --iterations 30000 --ckpt-every 1000 --log-every 500 --time-limit-hours 10.5 "
            f"--out {RUNS}/{name} --name {name} --results {RES}/{name}.md")

# resume: bring back checkpoints from an earlier version's output, if attached as an input
for jobs in JOBS.values():
    for j in jobs:
        prev = glob.glob(f"/kaggle/input/**/runs/{j['name']}/ckpt.pt", recursive=True)
        dst = os.path.join(RUNS, j["name"])
        if prev and not os.path.exists(os.path.join(dst, "ckpt.pt")):
            shutil.copytree(os.path.dirname(prev[0]), dst, dirs_exist_ok=True)
            print("resuming", j["name"], "from", prev[0])

import torch
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
chains = {g: " ; ".join(f"({job_cmd(j)})" for j in jobs) for g, jobs in JOBS.items()}
if n_gpu < 2:   # Colab: everything on one GPU, one job after the other
    chains = {0: " ; ".join(chains.values())}
procs = {}
for gpu, cmd in chains.items():
    log = open(os.path.join(RES, f"gpu{gpu}.log"), "w")
    print(f"GPU {gpu}:", cmd.replace(" ; ", "\n   "), "\n")
    procs[gpu] = subprocess.Popen(f"cd {REPO} && {cmd}", shell=True, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
t0 = time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(300)
    tails = {g: open(os.path.join(RES, f"gpu{g}.log")).read().strip().splitlines()[-1:] for g in procs}
    print(f"[{(time.time() - t0) / 3600:.2f} h]", {g: t[0][:110] if t else "" for g, t in tails.items()}, flush=True)
for g, p in procs.items():
    print(f"GPU {g} exit code {p.returncode}")
    print("".join(open(os.path.join(RES, f"gpu{g}.log")).readlines()[-15:]))
'''

PHASE4_PACKAGE = r'''
for jobs in JOBS.values():
    for j in jobs:
        d = os.path.join(RUNS, j["name"])
        for f in glob.glob(os.path.join(d, "log.csv")) + glob.glob(os.path.join(d, "test_000.png")):
            shutil.copy(f, os.path.join(RES, f"{j['name']}_{os.path.basename(f)}"))
sh(f"cd {REPO} && python bench/phase4_tables.py --runs {RES} --out {RES}/phase4_summary_{SESSION}.md")
shutil.make_archive(RES, "zip", RES)
print("download:", RES + ".zip   (final.pt files stay in runs/ in this version's output, for the viewer later)")
'''


def phase4_notebook(session, title, about, jobs, datasets, need_gsplat, est, extra=None):
    blender = sorted({j["scene"] for js in jobs.values() for j in js if j["dataset"] == "blender"})
    mip = sorted({j["scene"] for js in jobs.values() for j in js if j["dataset"] == "mipnerf360"})
    tt = sorted({j["scene"] for js in jobs.values() for j in js if j["dataset"] == "tandt"})
    get = []
    if blender:
        get.append(f'sh(f"cd {{REPO}} && python scripts/get_data.py blender --dest {{DATA}} --scenes {" ".join(blender)}")')
    if mip:
        get.append(f'sh(f"cd {{REPO}} && python scripts/get_data.py mipnerf360 --dest {{DATA}} --scenes {" ".join(mip)}")')
    if tt:
        get.append(f'sh(f"cd {{REPO}} && python scripts/get_data.py tandt --dest {{DATA}} --scenes {" ".join(tt)}")')
    cells = [
        md(f"""
# Phase 4, session {session}: {title}

{about}

Run on **Kaggle** with **GPU T4 x2** and Internet on: **Save Version → Save & Run All (Commit)**, then stop the draft session. With one GPU (Colab) the jobs run one after the other.

Every run: 30k iterations, paper hyperparameters, seed 0, the paper's evaluation protocol. Mip-NeRF 360 / T&T runs cap the Gaussian count at 3,000,000 to fit the T4's 15 GB (recorded in each results file). Checkpoints every 1000 iterations and a stop at 10.5 h: if the session ends early, attach this version's output as an input and run it again to resume.

**GPU-hour estimate (from the Phase-3 timings; a guess until measured):** {est}

At the end: download `phase4_{session}_results.zip` from the Output tab.
"""),
        code(SETUP),
        code(f'SESSION = "{session}"\nCAP = 3_000_000\nJOBS = {json.dumps(jobs, indent=1)}\nJOBS = {{int(k): v for k, v in JOBS.items()}}'),
        code(r'''
# all unit tests + GPU tests (Triton kernels vs the reference); stops the notebook if anything fails
sh(f"cd {REPO} && python -m pytest -p no:cacheprovider -rs tests")
'''),
        code("# datasets (into /tmp/data, not saved as output)\n" + "\n".join(get)),
    ]
    if need_gsplat:
        cells.append(code(GSPLAT_INSTALL))
    cells += [md("## Training jobs"), code(PHASE4_RUN)]
    if extra:
        cells += [md(extra[0]), code(extra[1])]
    cells += [md("## Package results"), code(PHASE4_PACKAGE)]
    return notebook(cells)


def j(name, who, dataset, scene):
    return {"name": name, "who": who, "dataset": dataset, "scene": scene}


def phase4_all():
    B = lambda s, w: j(f"{s}_{w}", w, "blender", s)
    M = lambda s, w: j(f"{s}_{w}", w, "mipnerf360", s)
    T = lambda s, w: j(f"{s}_{w}", w, "tandt", s)
    diag = ("## Diagnostic: my rasterizer vs gsplat on a trained scene (alpha clamp 0.99 vs 0.999)", r'''
sh(f"cd {REPO} && python bench/compare_gsplat.py --scene {RUNS}/lego_gsplat/final.pt --data {DATA}/nerf_synthetic/lego "
   f"--views 20 --out {RES}/compare_gsplat_lego.md")
''')
    return {
        "phase4_A_synthetic_mine": phase4_notebook(
            "A", "NeRF-synthetic, my Triton rasterizer (8 scenes)",
            "All 8 NeRF-synthetic scenes at 800x800 with my trainer + Triton rasterizer, 4 scenes per GPU. Includes the lego retrain after the density-control fix.",
            {0: [B(s, "mine") for s in ("lego", "ship", "drums", "materials")],
             1: [B(s, "mine") for s in ("chair", "ficus", "hotdog", "mic")]},
            ["blender"], False, "~1-2 h per scene, 4 per GPU in parallel: about 5-8 h of session time."),
        "phase4_B_synthetic_gsplat": phase4_notebook(
            "B", "NeRF-synthetic with gsplat's rasterizer (reference) + diagnostic",
            "The same 8 scenes, same training loop and settings, with gsplat's rasterizer (bench/train_gsplat.py), for the measured mine-vs-gsplat comparison. Then the alpha-clamp diagnostic on the trained lego.",
            {0: [B(s, "gsplat") for s in ("lego", "ship", "drums", "materials")],
             1: [B(s, "gsplat") for s in ("chair", "ficus", "hotdog", "mic")]},
            ["blender"], True, "gsplat build ~0.2 h; ~0.3-0.6 h per scene, 4 per GPU: about 1.5-3 h.", diag),
        "phase4_C_360_mine": phase4_notebook(
            "C", "Mip-NeRF 360 garden + bicycle, my Triton rasterizer",
            "garden on GPU 0, bicycle on GPU 1 (images_4, every 8th image held out, SfM init).",
            {0: [M("garden", "mine")], 1: [M("bicycle", "mine")]},
            ["mipnerf360"], False, "about 4-7 h (unknown until measured: these scenes reach millions of Gaussians)."),
        "phase4_D_360_tandt_gsplat": phase4_notebook(
            "D", "gsplat reference: garden, bicycle, kitchen, truck",
            "The same training loop with gsplat's rasterizer on the real scenes.",
            {0: [M("garden", "gsplat"), M("kitchen", "gsplat")], 1: [M("bicycle", "gsplat"), T("truck", "gsplat")]},
            ["mipnerf360", "tandt"], True, "about 2-4 h."),
        "phase4_E_kitchen_truck_mine": phase4_notebook(
            "E", "Mip-NeRF 360 kitchen (indoor) + Tanks & Temples truck, my Triton rasterizer",
            "kitchen (images_2) on GPU 0, truck on GPU 1.",
            {0: [M("kitchen", "mine")], 1: [T("truck", "mine")]},
            ["mipnerf360", "tandt"], False, "about 3-5 h."),
    }


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    books = {"phase2_lego": phase2(), "phase3_triton": phase3(), **phase4_all()}
    for name, nb in books.items():
        path = os.path.join(ROOT, "notebooks", f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path)
