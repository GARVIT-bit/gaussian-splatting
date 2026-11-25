# lego_triton_800: training with render mode `triton` (my Triton tile rasterizer, hand-written backward)

- Date: 2026-11-24  |  git commit: 85184dc
- Hardware: Tesla T4 (15 GB), CUDA 13.0  |  Python 3.13.16  |  PyTorch 2.11.0+cu130
- Data: `lego` (blender), 100 train / 200 test views at 800x800 (downscale 1), background [1.0, 1.0, 1.0]
- Init: 100000 random points in [-1.3, 1.3]^3, seed 0, 1 run
- Iterations: 30000  |  Gaussian cap: none  |  final Gaussians: 97405  |  SH degree: 3
- Training time (sum of iteration times, excl. loading/eval): 0.93 h (0.112 s/iteration)
- Peak GPU memory allocated: 4.76 GB
- Render time per test view (mean over all 200 test views, renders only, no grad): 10.6 ms (94.2 views/s)

| metric | value |
|---|---:|
| test PSNR (dB) | 32.46 |
| test SSIM | 0.9680 |
| test LPIPS | 0.0434 |
| test views | 200 |

PSNR per view: min 24.43, max 36.06, std 2.60. Evaluation took 80 s. LPIPS (VGG), lpips package

Full config:

```json
{
 "iterations": 30000,
 "sh_degree": 3,
 "sh_up_every": 1000,
 "position_lr_init": 0.00016,
 "position_lr_final": 1.6e-06,
 "position_lr_delay_mult": 0.01,
 "position_lr_delay_steps": 0,
 "position_lr_max_steps": 30000,
 "feature_lr": 0.0025,
 "opacity_lr": 0.025,
 "scaling_lr": 0.005,
 "rotation_lr": 0.001,
 "lambda_dssim": 0.2,
 "percent_dense": 0.01,
 "densify_from": 500,
 "densify_until": 15000,
 "densify_every": 100,
 "densify_grad_threshold": 0.0002,
 "opacity_reset_every": 3000,
 "min_opacity": 0.005,
 "max_screen_size": 20.0,
 "max_gaussians": 0,
 "init_opacity": 0.1,
 "seed": 0,
 "render_mode": "triton",
 "batch_elems": 4194304,
 "device": "cuda"
}
```
