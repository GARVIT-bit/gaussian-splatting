# GPU-hour budget

Quota: one Kaggle account, about **30 GPU-hours per week** (2 x T4 per session, 12 h max per session; Kaggle counts wall-clock time while the session is on, with both GPUs counting as one session). Fallback: Colab free T4 (no fixed quota, sessions get cut). Every estimate below is a **guess until measured**. The "measured" column gets filled from the notebook logs, and later rows are re-estimated from it.

## How the estimates are made

- **Paper reference point (reported, A6000 GPU):** 3DGS trains a Mip-NeRF 360 scene in about 35-45 min at 30k iterations and a NeRF-synthetic scene in a few minutes.
- **T4 vs A6000 (assumption):** about 3-4x slower (roughly 8 vs 39 FP32 TFLOPS, and 320 vs 768 GB/s memory bandwidth).
- **My Triton rasterizer vs gsplat (assumption):** 1.5-2.5x slower until tuned.
- **Phase 2 reference renderer (tiled PyTorch, autograd):** measured on a Kaggle T4 (2026-11-23, commit 93b96fd): 0.674 s/iteration at 200x200 with 100k Gaussians (300 iterations, peak 0.40 GB). Batches 4x bigger (`--batch-elems 2^24`) were no faster (0.690 s/it at iteration 100), so the default stays. At 100k Gaussians, 30k iterations project to ~5.4 h; densification up to the 200k cap will change that.
- **Overheads:** session setup + dataset download/unzip + evaluation + writing results: about 0.3 h per session.

## Planned runs

| # | Phase | Run | Settings | Est. GPU-h | Platform | Measured GPU-h |
|---|---|---|---|---:|---|---:|
| 1 | 2 | Debug: notebook smoke test, timing of the reference renderer | lego, 200x200, 300 iters | 0.3 | Kaggle | ~0.6 (interactive session, incl. a batch-size test) |
| 2 | 2 | Lego, reference renderer, low resolution (Phase 2 "done") | lego 200x200 (4x down), 30k iters, 100k random init, max 200k Gaussians | 3-8 | Kaggle GPU 0 | 4.0 (whole committed version incl. setup, tests and timing run; training itself 3.86 h) |
| 3 | 3 | Fast rasterizer GPU tests (forward + all gradients vs reference) | random scenes | 0.5 | Kaggle | _pending_ |
| 4 | 3 | Rasterizer benchmark vs reference and gsplat | 3 resolutions x 4 Gaussian counts | 0.5 | Kaggle | _pending_ |
| 5 | 3 | Lego full resolution with my rasterizer (Phase 3 "done") | 800x800, 30k | 0.5-1 | Kaggle | _pending_ |
| 6 | 4 | NeRF-synthetic, 8 scenes, mine | 800x800, 30k, white bg | 4-8 | Kaggle 2 GPUs in parallel | _pending_ |
| 7 | 4 | NeRF-synthetic, 8 scenes, gsplat (same settings, same T4) | 800x800, 30k | 2-4 | Kaggle 2 GPUs | _pending_ |
| 8 | 4 | Mip-NeRF 360 garden + bicycle, mine | images_4, 30k | 4-7 | Kaggle 2 GPUs | _pending_ |
| 9 | 4 | Mip-NeRF 360 garden + bicycle, gsplat | images_4, 30k | 2-3 | Kaggle 2 GPUs | _pending_ |
| 10 | 4 | Mip-NeRF 360 kitchen (indoor), mine + gsplat | images_2, 30k | 3-5 | Kaggle | _pending_ |
| 11 | 4 | Tanks & Temples truck, mine + gsplat (if time) | 3DGS release resolution, 30k | 2-4 | Kaggle | _pending_ |
| 12 | 4 | Own phone capture, mine (COLMAP on CPU in the session) | ~150 frames, 30k | 2-3 | Kaggle | _pending_ |
| 13 | 6 | Compression evaluation renders (each quantisation step) | held-out views of 2-3 scenes | 1 | Kaggle | _pending_ |
| | | **Total** | | **25-48** | | |

## Schedule (about 30 h/week)

- **Week 1:** runs 1-5 (Phases 2-3), about 5-11 h. The rest of the week stays spare for debugging reruns.
- **Week 2:** runs 6-9 (Phase 4, synthetic + 360 outdoor) on both GPUs, about 12-22 h. This is two or three sessions of under 12 h each.
- **Week 3:** runs 10-13, about 8-13 h.
- Colab free is the overflow for single-GPU runs (2, 5, 12) if Kaggle's quota runs out.

If measured timings push the total beyond what three to four weeks can hold, I stop and report before cutting anything (CLAUDE.md rule).

## Settings that apply to every run (recorded in each results file)

- Paper evaluation protocol: NeRF-synthetic at 800x800 with its own test split, white background. Mip-NeRF 360 with every 8th image held out, `images_4` for outdoor and `images_2` for indoor scenes. Tanks & Temples as in the 3DGS release.
- The Gaussian cap is stated per run. Phase-2 reference runs cap at 200k Gaussians because the tiled PyTorch compositing is slow.
