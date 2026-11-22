# 3D Gaussian Splatting from scratch

Rebuild a 3D scene from photos or a phone video and render it in the browser: my own differentiable Gaussian rasterizer (a pure-PyTorch reference, then a tile-based GPU version with a hand-written backward pass), the training loop with adaptive density control, evaluation on standard datasets, a WebGL viewer, and measured compression. Based on Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.

**Status:** Phase 1 of 6 done (maths + pure-PyTorch reference renderer, CPU). Phases 2-6 are not started; no GPU numbers exist yet.

| Phase | What | Status |
|---|---|---|
| 1 | Maths + pure-PyTorch reference renderer (laptop CPU) | done |
| 2 | Training pipeline (density control, loaders, checkpoints) | not started |
| 3 | Fast tile-based rasterizer with hand-written backward | not started |
| 4 | Real scenes + evaluation vs paper and gsplat | not started |
| 5 | WebGL viewer | not started |
| 6 | Compression (measured) + ship | not started |

## Setup (Windows laptop, CPU)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe numpy pillow pytest
.venv\Scripts\python.exe -m pytest
```

Demos:

```powershell
.venv\Scripts\python.exe scripts\fit_2d.py          # fit an image with 2D Gaussians -> docs/images/phase1_fit2d.png
.venv\Scripts\python.exe scripts\fit_synthetic.py   # fit a synthetic 3D scene -> bench/results/phase1_synthetic_fit.md
```

GPU training and evaluation (Phases 2-4) will run on Kaggle (2 x T4) or Colab through notebooks in `notebooks/`.

## Layout

```
splat/
  maths.py        quaternion -> rotation, Sigma = R S S^T R^T, 2x2 inverse (conic)
  camera.py       pinhole camera (OpenCV convention), look_at, orbit cameras
  projection.py   world -> camera, perspective Jacobian, EWA 2D covariance, conic, radius
  sh.py           real spherical harmonics, degree 0-3
  gaussians.py    the parameter set (means, log-scales, quats, opacity logits, SH dc / rest)
  reference.py    dense pure-PyTorch rasterizer (autograd gives the gradients)
  losses.py       L1, SSIM, (1-l) L1 + l D-SSIM, PSNR
  fit2d.py        2D Gaussian image fitting (warm-up)
  simple_fit.py   Phase-1 3D fit with a fixed set of Gaussians (no densification yet)
scripts/          demos that write images and bench/results/*.md
tests/            CPU unit tests (float64 where gradients are checked)
```

## How it works

### Phase 1: one Gaussian, from 3D to a pixel

**Parameters.** Each Gaussian has a mean μ ∈ ℝ³, a log-scale s ∈ ℝ³, a quaternion q = (w, x, y, z), an opacity logit, and SH colour coefficients. Everything is stored unconstrained so Adam can move it freely: scale = exp(s), opacity = sigmoid(logit), and q is normalised before use.

**3D covariance.** A covariance matrix must stay symmetric positive semi-definite while being optimised, so it is never optimised directly. It is built as

  Σ = R S Sᵀ Rᵀ,  R = rotation matrix of q/|q|,  S = diag(exp(s)).

Its eigenvectors are R's columns and its eigenvalues are the squared scales, so it is an oriented ellipsoid.

**To the camera.** With the world→camera rotation W and translation t, the centre in camera space is t_c = W μ + t, and the pixel position is u = f_x x/z + c_x, v = f_y y/z + c_y (OpenCV axes: x right, y down, z forward; pixel (i, j) has its centre at (i + ½, j + ½)).

**EWA projection.** Perspective projection is not linear, so a 3D Gaussian does not project to an exact 2D Gaussian. EWA splatting linearises the projection at the Gaussian's centre with its Jacobian

  J = ∂(u, v)/∂t_c = [[f_x/z, 0, −f_x x/z²], [0, f_y/z, −f_y y/z²]],

and a linear map of a Gaussian is a Gaussian, so the screen-space covariance is

  Σ' = J W Σ Wᵀ Jᵀ + 0.3 I.

The 0.3 I term (a ~0.55 px standard deviation blur) is the paper's low-pass filter: it keeps every splat at least about one pixel wide so tiny Gaussians do not alias or vanish between pixel centres. As in the reference CUDA code, x/z and y/z are clamped to 1.3 × the half-FoV tangent inside J only, so Gaussians far outside the view do not get exploding Jacobians. A Monte-Carlo test checks the approximation: 400k samples of a small 3D Gaussian, projected exactly, have a 2D covariance within 2 % of Σ' (without the low-pass).

**Conic and radius.** The inverse Σ'⁻¹ = [[a, b], [b, c]] (the "conic") is what the rasterizer needs: the Gaussian's falloff at offset d = (dx, dy) from its 2D centre is exp(−½(a dx² + 2b dx dy + c dy²)). For culling, the radius is 3 standard deviations along the largest axis, r = ⌈3 √λ_max⌉ with λ_max = m + √(m² − det), m = (Σ'₁₁ + Σ'₂₂)/2. Gaussians with z ≤ 0.2 (the near plane) or det ≤ 0 are dropped.

**Colour from spherical harmonics.** Colour depends on the view direction d = (μ − camera centre)/|…|. It is a weighted sum of real SH basis functions up to degree 3 (1, 4, 9 or 16 coefficients per colour channel): colour = Σ_k c_k Y_k(d) + 0.5, clamped at 0. Degree 0 is a constant (diffuse) colour; higher degrees add view-dependent effects such as highlights. The +0.5 offset makes a zero DC coefficient mean mid-grey. The basis constants are checked against the textbook formula built from associated Legendre polynomials and against orthonormality on the sphere.

**Alpha compositing.** Gaussians are sorted by camera-space depth of their centre (one global sort per image, not per pixel) and blended front to back at each pixel:

  α_i = min(0.99, o_i · exp(−½ dᵀ Σ'⁻¹ d)),  T_i = Π_{j<i} (1 − α_j),  C = Σ_i α_i T_i c_i + T_final · background.

T_i is the transmittance (how much light from behind still gets through). Three rules copied from the CUDA kernel matter for matching it later: α is clamped to 0.99 (so T never hits exactly 0, which keeps the backward pass invertible), contributions with α < 1/255 are skipped, and blending stops at the first Gaussian that would push T below 10⁻⁴ (that Gaussian is not blended). The reference renderer does this densely: every pixel against every visible Gaussian, with `cumprod` for the transmittances and a prefix mask for the early stop. A test checks it against a literal per-pixel loop written like the CUDA kernel.

**Gradients.** In Phase 1 PyTorch autograd differentiates through all of the above. `torch.autograd.gradcheck` in float64 checks the gradients of the rendered image with respect to every parameter (means, log-scales, quaternions, opacities, SH DC and higher orders) against finite differences. These are the gradients the hand-written backward pass of Phase 3 has to reproduce.

**Fitting.** Two demos show the gradients are useful, not just correct: 200 2D Gaussians fitting a procedural image, and a fixed set of 3D Gaussians fitting a synthetic 3D Gaussian scene from orbit views, scored on held-out views (Adam, 0.8 L1 + 0.2 D-SSIM). Both are also tests with convergence thresholds.

## Results

### Phase 1: synthetic scene fit (CPU, reference renderer)

From `bench/results/phase1_synthetic_fit.md` (`scripts/fit_synthetic.py`):

Setup: 40 random ground-truth Gaussians (SH degree 1), 24 orbit views at 48×48 px with every 8th view held out (3 test views), a model of 200 randomly initialised Gaussians without densification, 2,000 Adam iterations, i5-1135G7 CPU, PyTorch 2.15 CPU, commit 7a071a7.

| seed | test PSNR at init (dB) | test PSNR final (dB) | test SSIM final | wall time (s) |
|---:|---:|---:|---:|---:|
| 0 | 13.79 | 42.67 | 0.9971 | 138 |
| 1 | 12.96 | 44.42 | 0.9976 | 112 |
| 2 | 13.10 | 42.18 | 0.9965 | 119 |

Mean held-out PSNR **43.09 dB** (range 42.18-44.42) and mean SSIM 0.9971 over 3 runs. This is a synthetic check that the optimisation works. It is not comparable to the paper's numbers on real data.

Init | fitted | ground truth on a held-out view (48×48, upscaled 4×):

![phase 1 synthetic fit](docs/images/phase1_synthetic_fit.png)

2D warm-up, target | fit (`scripts/fit_2d.py`):

![phase 1 2D fit](docs/images/phase1_fit2d.png)

## Bugs found on the way

1. **A float64 test leaked into the float32 tests.** *Symptom:* `test_fit.py` passed on its own but failed in the full suite with `expected m1 and m2 to have the same dtype`. *Cause:* the float64 test modules called `torch.set_default_dtype(torch.float64)` at import time. pytest imports every module before running any test, so the global default was float64 everywhere, the Gaussians were created in float64 and the cameras in float32. *Fix:* a `float64` fixture in `tests/conftest.py` that sets and restores the default per test, and `render()` now casts the camera to the Gaussians' dtype. *Guard:* the full suite runs the float32 fit tests next to the float64 ones.
2. **A test sat exactly on the early-stop threshold.** *Symptom:* the early-termination test expected 4 layers with α = 0.9 to be blended, but the renderer blended 3. *Cause:* after 4 layers T = 0.1⁴, which is 10⁻⁴ only in exact arithmetic; in float64 1 − 0.9 = 0.09999999999999998 and the product is 9.999999999999991 × 10⁻⁵, just below 10⁻⁴, so the "stop if T < 10⁻⁴" rule fires one layer early. The renderer was right; the test was ambiguous. *Fix:* the test uses α = 0.85 (T after 4 layers = 5.1 × 10⁻⁴, after 5 = 7.6 × 10⁻⁵), far from the threshold. *Lesson for Phase 3:* float32 GPU kernels and the reference can disagree about pixels whose T is within rounding of 10⁻⁴; the comparison tests must expect that.
3. **Design mistake caught before running (no test found it):** I first gave the higher-order SH coefficients a smaller learning rate by scaling their gradients with a hook. Adam divides by the running RMS of the gradient, so a constant gradient scale cancels out and changes nothing. *Fix:* the SH DC and higher-order coefficients are separate tensors (`sh_dc`, `sh_rest`) in separate Adam parameter groups, as in the original implementation.

## Honest limits (Phase 1)

- The reference renderer is dense (every pixel × every Gaussian), so it only runs at tiny resolutions and Gaussian counts. It is the correctness baseline, not the fast path; the tile-based rasterizer is Phase 3.
- No adaptive density control (clone / split / prune / opacity reset) yet: the Phase-1 fit uses a fixed number of Gaussians. That and the real data loaders are Phase 2.
- The Phase-1 scene is synthetic (random ground-truth Gaussians rendered by the same renderer), so it shows the optimisation works, not that it reconstructs real photos.
- The EWA projection is a first-order approximation (exact only for small Gaussians); this is the paper's method, not a bug. Anti-aliasing beyond the 0.3 low-pass (Mip-Splatting) is a stretch goal.
- Camera poses and initial points for real scenes will come from COLMAP, which is not my code. Structure-from-motion is a separate project.

## References

Kerbl et al., 3D Gaussian Splatting (SIGGRAPH 2023) and its supplementary; Zwicker et al., EWA Splatting (2002); the original `diff-gaussian-rasterization` CUDA code and gsplat's maths notes (read for reference, no code vendored).
