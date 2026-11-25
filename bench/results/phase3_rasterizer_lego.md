# Phase 3: rasterizer benchmark (one render: forward, and forward + backward)

- Date: 2026-11-24  |  git commit: 85184dc
- GPU: Tesla T4  |  PyTorch 2.11.0+cu130  |  CUDA 13.0  |  Triton 3.6.0  |  Python 3.13.16
- Reference comparison: gsplat 1.5.3 (rasterize_mode classic, eps2d 0.3, near plane 0.2, tile 16)
- Scene: trained scene /kaggle/working/runs/lego_triton_800/final.pt, SH degree 3, white background
- Camera: first test camera of lego
- Timing: median of 20 (reference: 4) after warm-up, wall clock with cuda.synchronize, includes preprocess + binning + sort. Loss for backward: image.sum().

| resolution | scene | Gaussians | tile pairs | mine (Triton) fwd ms | mine fwd+bwd ms | reference fwd+bwd ms | gsplat fwd ms | gsplat fwd+bwd ms | speed-up vs reference | mine / gsplat (fwd+bwd) | peak MB (mine) | max abs diff mine vs ref | max abs diff mine vs gsplat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200x200 | trained | 97405 | 208450 | 12.5 | 82.8 (82.1-84.5) | 361.5 | 2.9 | 7.6 (7.2-8.6) | 4.4x | 10.86x | 892 | 5.1e-05 | 8.6e-02 |
| 400x400 | trained | 97405 | 283625 | 16.0 | 50.0 (49.3-50.7) | 481.4 | 2.3 | 5.9 (5.6-6.4) | 9.6x | 8.52x | 894 | 1.4e-03 | 8.5e-02 |
| 800x800 | trained | 97405 | 536194 | 16.4 | 42.4 (41.5-44.1) | 850.5 | 2.8 | 8.3 (8.0-8.8) | 20.0x | 5.12x | 910 | 1.2e-06 | 9.2e-02 |

"mine / gsplat" above 1 means mine is slower. Reference rows are skipped where one render takes over 4 s. Forward+backward cells show median (min-max) over the reps.
