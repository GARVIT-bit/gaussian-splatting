# Phase 3: rasterizer benchmark (one render: forward, and forward + backward)

- Date: 2026-11-24  |  git commit: 85184dc
- GPU: Tesla T4  |  PyTorch 2.11.0+cu130  |  CUDA 13.0  |  Triton 3.6.0  |  Python 3.13.16
- Reference comparison: gsplat 1.5.3 (rasterize_mode classic, eps2d 0.3, near plane 0.2, tile 16)
- Scene: random Gaussians in a unit ball, scales 0.004-0.03, opacity 0.1-0.9, SH degree 3, white background
- Camera: distance 3.2, FoV 50 deg
- Timing: median of 20 (reference: 4) after warm-up, wall clock with cuda.synchronize, includes preprocess + binning + sort. Loss for backward: image.sum().

| resolution | scene | Gaussians | tile pairs | mine (Triton) fwd ms | mine fwd+bwd ms | reference fwd+bwd ms | gsplat fwd ms | gsplat fwd+bwd ms | speed-up vs reference | mine / gsplat (fwd+bwd) | peak MB (mine) | max abs diff mine vs ref | max abs diff mine vs gsplat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200x200 | random 10000 | 10000 | 24915 | 6.5 | 16.1 (12.3-16.9) | 72.6 | 1.3 | 2.6 (2.3-3.0) | 4.5x | 6.28x | 207 | 2.4e-07 | 1.2e-03 |
| 200x200 | random 50000 | 50000 | 123499 | 6.7 | 25.8 (25.3-26.0) | 226.9 | 1.8 | 3.5 (3.4-4.3) | 8.8x | 7.33x | 552 | 2.4e-07 | 1.8e-05 |
| 200x200 | random 100000 | 100000 | 247817 | 8.6 | 39.5 (38.6-42.5) | 406.2 | 1.9 | 4.4 (4.2-4.5) | 10.3x | 9.00x | 1003 | 4.2e-07 | 6.5e-04 |
| 200x200 | random 300000 | 300000 | 742135 | 25.1 | 99.6 (98.7-100.2) | 1100.4 | 4.3 | 9.9 (9.8-10.2) | 11.0x | 10.05x | 2679 | 5.4e-07 | 4.5e-04 |
| 400x400 | random 10000 | 10000 | 39862 | 5.6 | 13.2 (12.1-15.7) | 97.9 | 1.2 | 2.4 (2.1-2.6) | 7.4x | 5.51x | 210 | 3.0e-07 | 8.9e-04 |
| 400x400 | random 50000 | 50000 | 199220 | 6.4 | 20.8 (20.2-21.6) | 331.7 | 1.7 | 3.7 (3.7-4.0) | 15.9x | 5.56x | 559 | 4.2e-07 | 1.2e-03 |
| 400x400 | random 100000 | 100000 | 399820 | 8.2 | 31.1 (30.6-31.4) | 602.6 | 2.0 | 4.9 (4.7-5.1) | 19.4x | 6.31x | 1008 | 3.6e-07 | 1.1e-03 |
| 400x400 | random 300000 | 300000 | 1198095 | 23.2 | 80.4 (79.4-81.2) | 1795.8 | 4.2 | 10.1 (10.0-10.5) | 22.3x | 7.95x | 2689 | 4.2e-07 | 1.4e-03 |
| 800x800 | random 10000 | 10000 | 84637 | 6.4 | 12.0 (11.7-15.9) | 178.3 | 1.5 | 2.7 (2.6-3.1) | 14.8x | 4.46x | 229 | 4.8e-07 | 2.7e-03 |
| 800x800 | random 50000 | 50000 | 423732 | 7.6 | 24.8 (24.2-25.5) | 627.5 | 2.2 | 6.5 (6.1-6.9) | 25.3x | 3.80x | 580 | 4.8e-07 | 3.6e-03 |
| 800x800 | random 100000 | 100000 | 848531 | 10.6 | 37.5 (37.2-39.5) | 1212.6 | 2.8 | 8.2 (7.9-8.7) | 32.3x | 4.58x | 1034 | 2.3e-03 | 5.6e-03 |
| 800x800 | random 300000 | 300000 | 2546509 | 27.9 | 87.2 (86.3-87.8) | 3572.2 | 5.5 | 13.9 (13.6-14.3) | 41.0x | 6.29x | 2732 | 4.2e-07 | 6.3e-03 |

"mine / gsplat" above 1 means mine is slower. Reference rows are skipped where one render takes over 4 s. Forward+backward cells show median (min-max) over the reps.
