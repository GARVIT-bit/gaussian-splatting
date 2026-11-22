# 3D Gaussian Splatting from scratch: context for Claude Code

## What this project is
Project 5 of my resume set (I'm Garvit, CSE student, GitHub GARVIT-bit; I did computer vision work in my Hyprix internship). Done so far: ../mini-perplexity, ../crdt-editor, mini-docker-k8s (in WSL), ../llm-inference-engine. Don't touch those folders or their Docker containers. (../agentops was an abandoned plan; ignore it.)

Rebuild a real 3D scene from photos or a phone video and render it in real time in a browser, with my own implementation of 3D Gaussian Splatting (Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023):
- my own differentiable Gaussian rasterizer: a pure-PyTorch reference, then a fast tile-based GPU version with a hand-written backward pass;
- the training loop with spherical-harmonic colour, L1 + D-SSIM loss and adaptive density control (clone, split, prune, opacity reset);
- evaluation on standard datasets (PSNR / SSIM / LPIPS) compared honestly against the paper and against gsplat;
- my own scene captured on my phone;
- my own WebGL viewer that streams and renders the trained scene in the browser;
- one measured extension (compression, see Phase 6).

References to learn from (don't vendor their code): the 3DGS paper and its supplementary (EWA projection, the backward pass), the original `diff-gaussian-rasterization` CUDA code, gsplat (nerfstudio) and its paper/maths notes, antimatter15/splat (a minimal WebGL splat viewer), Mip-Splatting (anti-aliasing), and compact-3DGS papers for Phase 6.

## Rules (what must be my code)
- Mine: Gaussian parameterisation, projection (covariance from scale + quaternion, Jacobian of the perspective projection, 2D conic), SH evaluation, tile binning and depth sorting, alpha compositing, **the backward pass**, the optimiser setup and density control, the training loop, evaluation scripts, the viewer.
- Allowed tools: PyTorch (tensors, autograd for the reference renderer, Adam), Triton (or a CUDA extension via torch.utils.cpp_extension) for the fast rasterizer, COLMAP for camera poses and the initial sparse point cloud (structure-from-motion is a separate project: say so in the README), ffmpeg for video frames, the `lpips` package for the metric only, pycolmap or my own reader for COLMAP's files.
- gsplat / diff-gaussian-rasterization may be installed **only** in a separate benchmark script as a reference to compare speed and quality against. Never imported by my renderer or trainer.
- The fast rasterizer is only trusted after it matches the pure-PyTorch reference: forward images and every gradient (means, scales, rotations, opacities, SH coefficients) within tolerance, tested.

## Machine and environment
- Windows laptop, i5-1135G7, no NVIDIA GPU, 8 GB RAM, LOW disk. Python 3.12 venv at `.venv` (`.venv\Scripts\python.exe`), CPU-only PyTorch on the laptop. Node for the viewer.
- The laptop runs: the reference renderer at tiny resolutions, all CPU unit tests (maths, SH, projection, sorting logic, gradient checks in float64 on tiny scenes), data tooling, the viewer.
- GPU work runs on **Kaggle Notebooks first** (free, 2 x T4 16 GB, ~30 GPU-hours/week shown in my account, 12-hour sessions, "Save Version" runs headless in the background, outputs persist), with **Colab free (T4) as the fallback**. Notebooks in `notebooks/` must work on both: detect the platform, clone the repo from GitHub (Kaggle needs Internet switched on), install deps, run the GPU tests, train, evaluate, and write results (bench/results/*.md, small checkpoints, renders) as downloadable outputs for me to commit. On Kaggle's 2 x T4, train two scenes in parallel (one process per GPU) rather than splitting one model across GPUs. Before each GPU step, estimate the GPU-hours it needs so I can budget the weekly quota. Like Project 4: leave GPU numbers as placeholders until I've run the notebook. Sessions can end: training must checkpoint and resume.
- T4 has 16 GB: use downsampled images for real scenes (e.g. 4x for Mip-NeRF 360) and cap the number of Gaussians; record the settings in every results file.
- Datasets are big: download them in the Kaggle/Colab session (or attach them as a Kaggle Dataset once, so later sessions don't re-download), never on the laptop. Only small exported scenes (a few MB to tens of MB) come to the laptop for the viewer.
- Ports in use on this machine: 8080, 8090, 7070, 8001, 27017, 27018, 6379, 6380. Use 5176 for the viewer dev server.
- No API keys needed for anything.
- **Everything stays free, and the experiment stays full-size.** Only free resources: my one Kaggle account (~30 GPU-hours/week), Colab's free tier, Lightning AI's free tier if I set it up, free public datasets, GitHub Pages. No paid GPUs or credits.
- Do NOT shrink the final experiment (scenes, resolution, iterations) to save quota. Instead:
  - Use the paper's own evaluation protocol (its standard image downscaling per dataset), so results are comparable to the paper.
  - Debug on tiny runs (few hundred iterations, small images) and on the laptop CPU where possible; spend GPU hours only on runs that produce reported numbers.
  - Write a GPU-hour budget at the start of Phase 2 (every planned run x estimated hours) and spread it over the weeks and platforms; update it as real timings come in.
  - Checkpoint often so a dropped session never wastes a run; stop sessions as soon as runs finish (Kaggle counts wall-clock time).
  - If the budget really can't fit, tell me and wait; never cut silently.

## Build phases
1. **Maths + pure-PyTorch reference renderer (laptop).** Gaussians with mean, log-scale, quaternion, logit-opacity and SH coefficients; Σ = R S Sᵀ Rᵀ; world → camera; EWA projection Σ' = J W Σ Wᵀ Jᵀ (+0.3 low-pass on the diagonal); 2D conic and pixel radius (3σ); SH degree 0–3 evaluated in the view direction; front-to-back alpha compositing with early stop at transmittance < 1e-4 and alpha clamped to 0.99. Tests: projection vs finite differences and a brute-force per-pixel Gaussian evaluation; SH vs a known formula; compositing order; a tiny "fit one image with 2D Gaussians" demo that converges. Done when an autograd-trained reference fits a small synthetic scene on the CPU.
2. **Training pipeline (Colab, NeRF-synthetic "lego" first, then a few more).** Data loaders for Blender (NeRF-synthetic) and COLMAP formats; random init in a box for synthetic scenes, SfM points for real ones; losses (1 − λ)·L1 + λ·D-SSIM with λ = 0.2; Adam with per-parameter learning rates and the exponentially decayed position LR from the paper; SH degree raised every 1,000 iterations; adaptive density control (accumulate view-space positional gradient, clone small / split large Gaussians above the threshold, prune low opacity and huge ones, reset opacity every 3,000 iterations) on the paper's schedule; checkpoint + resume. Done when lego trains end to end with the reference renderer at low resolution and the test PSNR is reported (even if slow).
3. **Fast tile-based rasterizer (Colab).** Preprocess kernel (project, conic, radius, SH → colour, tiles touched); duplicate each Gaussian per tile with a 64-bit key (tile id << 32 | depth bits) and sort once; per-tile ranges; render kernel with 16 × 16 tiles front to back; **hand-written backward** that walks the same list back to front, reconstructing transmittance, and accumulates gradients for 2D means, conics, opacities and colours, then a backward preprocess to 3D means, scales, rotations and SH. Tests: forward and all gradients vs the reference on random scenes (float32 tolerances stated), empty tiles, Gaussians behind the camera, huge Gaussians. Bench: images/s and training iterations/s vs the reference and vs gsplat (reference only) at several resolutions and Gaussian counts. Done when it matches the reference and trains lego at full resolution.
4. **Real scenes + evaluation (Colab).** Mip-NeRF 360 (at least 2 scenes, e.g. garden and bicycle or kitchen), Tanks & Temples "truck" or "train" if time allows, NeRF-synthetic (all or several of the 8 scenes). Standard split: every 8th image held out for real scenes; NeRF-synthetic's own test split. Report PSNR, SSIM, LPIPS (VGG, state which), training time, peak GPU memory, number of Gaussians, render FPS, all at stated resolutions. Compare against the paper's reported numbers (label them "reported in the paper, different hardware and resolution") and against gsplat trained with the same settings on the same T4 (measured). Also my own capture: a phone video of an object or room, frames via ffmpeg, COLMAP poses, trained the same way; held-out views for metrics.
5. **WebGL viewer (laptop).** Export trained scenes to a compact binary format (position, scale, rotation, opacity, colour; SH degree 0 first, higher degrees optional). Viewer (Vite + TypeScript, React only if it helps the UI): WebGL2, one instanced quad per Gaussian, the 2D covariance computed in the vertex shader, Gaussian falloff + alpha blending in the fragment shader, depth sort in a Web Worker (counting sort on 16-bit depth keys), re-sort only when the camera moves enough, orbit + fly controls, progressive loading. Measure FPS on my laptop for each scene and Gaussian count. Deploy as a static site (GitHub Pages) with one or two small scenes.
6. **Measured extension: compression (and ship).** Shrink the exported scene: quantise positions (16-bit), scales/rotations (8-bit), colours, drop or reduce higher-order SH, prune low-contribution Gaussians, optionally codebook SH via k-means. Report size (MB) vs PSNR/SSIM/LPIPS on the held-out views for each step, so the trade-off is a curve, not a claim. Then README: architecture diagram, results tables, render images and a GIF, the honest-limits section, links to the live viewer.
   (Stretch, only if everything above is done: Mip-Splatting's 3D smoothing filter and 2D Mip filter for zoom-out aliasing, measured at multiple resolutions.)

## Lessons from my last four projects (follow these)
- Every number in the README comes from a script in bench/ (or a notebook cell that writes bench/results/<name>.md) with: GPU/CPU, PyTorch/Triton/CUDA versions, dataset and scene, image resolution, iterations, number of Gaussians, seed, git commit, date, number of runs and spread.
- Comparisons fair and stated: same scenes, same resolution, same iterations, same GPU for my code vs gsplat. Paper numbers are labelled as reported, never mixed into "measured".
- Rerun benchmarks after code changes before shipping. No stale numbers.
- No unmeasured claims anywhere, including docstrings, notebook text and demo comments ("real-time" only with a measured FPS next to it).
- When a test or benchmark finds a bug, write it up in the README ("bug found on the way": symptom, cause, fix, guarding test). Those are my best interview stories.
- Keep an "Honest limits" section in sync with the code (COLMAP not mine, T4 and downsampled resolutions, which features of the paper are missing, viewer limitations).

## Git
- `git init`; .gitignore for .venv, node_modules, datasets, checkpoints, large renders (commit only small result images, GIFs and small exported demo scenes, or use GitHub Releases for scene files). Commit at the end of each phase.
- Commits use my GitHub no-reply address as author and no co-author lines.
- Don't push until I say so. Repo will be GARVIT-bit/gaussian-splatting (Colab notebooks clone it, so I'll push before the first GPU phase).

## How I want to work
- Build without long explanations; tests after every module, run them before moving on; short summary at the end of each phase.
- Never drop or weaken tests to go faster.
- I need to understand what I ship: a short "How it works" README section per phase, including the maths (projection Jacobian, conic, compositing, the backward pass) in plain words with the key equations.
- Tell me exactly what to do on my side at each GPU step (which notebook, runtime type, what to download and where to put it).
- At the end, tell me it's done so my interview notes can be updated from the README and bench results.
