# Phase 1: synthetic scene fit with the autograd reference renderer (CPU)

- Date: 2026-11-23  |  git commit: 7a071a7
- CPU: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel  |  threads: 8  |  Python 3.12.12  |  PyTorch 2.15.0+cpu
- Scene: 40 random ground-truth Gaussians (SH degree 1) in [-0.7, 0.7]^3, black background
- Cameras: 24 orbit views at radius 3, FoV 50 deg, 48x48 px; every 8th view held out (3 test views)
- Model: 200 Gaussians, random init in [-0.9, 0.9]^3, grey, opacity 0.1; no densification (Phase 2)
- Optimiser: Adam eps 1e-15, LRs {'means': 0.004, 'log_scales': 0.01, 'quats': 0.005, 'opacity_logits': 0.05, 'sh_dc': 0.01, 'sh_rest': 0.0005}, loss 0.8 L1 + 0.2 D-SSIM, 2000 iterations, one view per iteration
- Seeds: [0, 1, 2] (3 runs)

| seed | test PSNR at init (dB) | test PSNR final (dB) | test SSIM final | wall time (s) |
|---:|---:|---:|---:|---:|
| 0 | 13.79 | 42.67 | 0.9971 | 138 |
| 1 | 12.96 | 44.42 | 0.9976 | 112 |
| 2 | 13.10 | 42.18 | 0.9965 | 119 |

**Mean test PSNR 43.09 dB (min 42.18, max 44.42); mean SSIM 0.9971** over 3 runs.

Test PSNR during training (seed 0):

| iteration | train loss | test PSNR (dB) |
|---:|---:|---:|
| 200 | 0.0190 | 31.88 |
| 400 | 0.0072 | 36.61 |
| 600 | 0.0068 | 37.96 |
| 800 | 0.0047 | 40.28 |
| 1000 | 0.0038 | 41.36 |
| 1200 | 0.0051 | 41.59 |
| 1400 | 0.0052 | 42.13 |
| 1600 | 0.0037 | 42.05 |
| 1800 | 0.0029 | 43.48 |
| 2000 | 0.0025 | 42.67 |

Image: docs/images/phase1_synthetic_fit.png (init | fitted | ground truth, held-out view, seed 0, nearest-upsampled 4x).
