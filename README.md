# 3D Gaussian Splatting from scratch

Rebuild a 3D scene from photos or a phone video and render it in real time in the browser: my own differentiable Gaussian rasterizer (reference + fast tile-based GPU version with a hand-written backward pass), the training loop with adaptive density control, evaluation on standard datasets, a WebGL viewer, and measured compression.

**Status:** not started. See `CLAUDE.md` for the plan.

## Setup (Windows laptop)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

GPU training and evaluation run on a free Colab T4 through the notebooks in `notebooks/`.
