# lego_triton_200: training with render mode `triton` (my Triton tile rasterizer, hand-written backward)

- Date: 2026-11-24  |  git commit: 85184dc
- Hardware: Tesla T4 (15 GB), CUDA 13.0  |  Python 3.13.16  |  PyTorch 2.11.0+cu130
- Data: `lego` (blender), 100 train / 200 test views at 200x200 (downscale 4), background [1.0, 1.0, 1.0]
- Init: 100000 random points in [-1.3, 1.3]^3, seed 0, 1 run
- Iterations: 30000  |  Gaussian cap: 200000  |  final Gaussians: 109205  |  SH degree: 3
- Training time (sum of iteration times, excl. loading/eval): 0.76 h (0.091 s/iteration)
- Peak GPU memory allocated: 4.82 GB
- Render time per test view (mean over all 200 test views, renders only, no grad): 13.3 ms (75.0 views/s)

| metric | value |
|---|---:|
| test PSNR (dB) | 35.43 |
| test SSIM | 0.9898 |
| test LPIPS | 0.0107 |
| test views | 200 |

PSNR per view: min 25.26, max 40.99, std 3.20. Evaluation took 8 s. LPIPS (VGG), lpips package

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
 "max_gaussians": 200000,
 "init_opacity": 0.1,
 "seed": 0,
 "render_mode": "triton",
 "batch_elems": 4194304,
 "device": "cuda"
}
```
