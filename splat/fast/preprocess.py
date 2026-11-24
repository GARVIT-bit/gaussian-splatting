"""Per-Gaussian preprocess with a hand-written backward pass (no autograd inside).

Forward, per Gaussian (same maths as splat.projection / splat.sh):
    q^ = q/|q|, R = R(q^), s = exp(log_s), M = R diag(s), Sigma = M M^T
    t = W mu + t_cam,   mu2d = (fx x/z + cx, fy y/z + cy)
    J = [[fx/z, 0, -fx tx/z^2], [0, fy/z, -fy ty/z^2]]  (tx = clamp(x/z) z, same for ty)
    T = J W,  Sigma' = T Sigma T^T + 0.3 I = [[A, B], [B, C]],  conic = (C, -B, A)/det
    o = sigmoid(logit),  d = (mu - cam)/|mu - cam|,  colour = max(sum_k Y_k(d) c_k + 0.5, 0)

Backward: the chain rule written out by hand, stage by stage, in reverse.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..camera import Camera
from ..projection import FRUSTUM_SLACK, LOW_PASS, NEAR_PLANE
from ..sh import C1, C2, C3, num_sh_coeffs, sh_basis


# ------------------------------------------------------------------ SH Jacobian

def sh_basis_jacobian(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """d Y_k / d dir for the polynomials in splat.sh: (N, 3) -> (N, K, 3)."""
    x, y, z = dirs.unbind(-1)
    zero = torch.zeros_like(x)
    rows = [torch.stack([zero, zero, zero], -1)]
    if degree >= 1:
        rows += [torch.stack([zero, -C1 + zero, zero], -1),
                 torch.stack([zero, zero, C1 + zero], -1),
                 torch.stack([-C1 + zero, zero, zero], -1)]
    if degree >= 2:
        rows += [C2[0] * torch.stack([y, x, zero], -1),
                 C2[1] * torch.stack([zero, z, y], -1),
                 C2[2] * torch.stack([-2 * x, -2 * y, 4 * z], -1),
                 C2[3] * torch.stack([z, zero, x], -1),
                 C2[4] * torch.stack([2 * x, -2 * y, zero], -1)]
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        rows += [C3[0] * torch.stack([6 * x * y, 3 * xx - 3 * yy, zero], -1),
                 C3[1] * torch.stack([y * z, x * z, x * y], -1),
                 C3[2] * torch.stack([-2 * x * y, 4 * zz - xx - 3 * yy, 8 * y * z], -1),
                 C3[3] * torch.stack([-6 * x * z, -6 * y * z, 6 * zz - 3 * xx - 3 * yy], -1),
                 C3[4] * torch.stack([4 * zz - 3 * xx - yy, -2 * x * y, 8 * x * z], -1),
                 C3[5] * torch.stack([2 * x * z, -2 * y * z, xx - yy], -1),
                 C3[6] * torch.stack([3 * xx - 3 * yy, -6 * x * y, zero], -1)]
    return torch.stack(rows, 1)


# ------------------------------------------------------------- quaternion maths

def quat_rotmat(qn: torch.Tensor) -> torch.Tensor:
    w, x, y, z = qn.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def rotmat_backward(qn: torch.Tensor, gR: torch.Tensor) -> torch.Tensor:
    """dL/dq^ from dL/dR, differentiating each entry of R(q^) by hand."""
    w, x, y, z = qn.unbind(-1)
    g = lambda i, j: gR[:, i, j]
    gw = 2 * (-z * g(0, 1) + y * g(0, 2) + z * g(1, 0) - x * g(1, 2) - y * g(2, 0) + x * g(2, 1))
    gx = 2 * (y * g(0, 1) + z * g(0, 2) + y * g(1, 0) - 2 * x * g(1, 1) - w * g(1, 2)
              + z * g(2, 0) + w * g(2, 1) - 2 * x * g(2, 2))
    gy = 2 * (-2 * y * g(0, 0) + x * g(0, 1) + w * g(0, 2) + x * g(1, 0) + z * g(1, 2)
              - w * g(2, 0) + z * g(2, 1) - 2 * y * g(2, 2))
    gz = 2 * (-2 * z * g(0, 0) - w * g(0, 1) + x * g(0, 2) + w * g(1, 0) - 2 * z * g(1, 1)
              + y * g(1, 2) + x * g(2, 0) + y * g(2, 1))
    return torch.stack([gw, gx, gy, gz], -1)


# -------------------------------------------------------------------- forward

@dataclass
class Saved:
    """Intermediates the backward pass needs."""
    qn: torch.Tensor
    qnorm: torch.Tensor
    R: torch.Tensor
    s: torch.Tensor
    M: torch.Tensor
    Sigma: torch.Tensor
    t: torch.Tensor
    tx_clamped: torch.Tensor
    ty_clamped: torch.Tensor
    J: torch.Tensor
    T: torch.Tensor
    cov2d: torch.Tensor
    det: torch.Tensor
    opac: torch.Tensor
    dirs: torch.Tensor
    vnorm: torch.Tensor
    basis: torch.Tensor
    col_raw: torch.Tensor
    valid: torch.Tensor


def preprocess_forward(means, log_scales, quats, opacity_logits, sh, cam: Camera, sh_degree: int):
    """Returns (means2d, conics, opacities, colors, depths, radii, valid) and the saved state."""
    W, tc = cam.R, cam.t
    qnorm = quats.norm(dim=-1, keepdim=True)
    qn = quats / qnorm
    R = quat_rotmat(qn)
    s = torch.exp(log_scales)
    M = R * s[:, None, :]
    Sigma = M @ M.transpose(-1, -2)

    t_raw = means @ W.T + tc
    valid = t_raw[:, 2] > NEAR_PLANE
    t = torch.where(valid[:, None], t_raw, torch.ones_like(t_raw))
    x, y, z = t.unbind(-1)
    means2d = torch.stack([cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy], -1)

    limx, limy = FRUSTUM_SLACK * cam.tan_fovx, FRUSTUM_SLACK * cam.tan_fovy
    ux, uy = x / z, y / z
    tx_clamped = (ux < -limx) | (ux > limx)
    ty_clamped = (uy < -limy) | (uy > limy)
    tx = ux.clamp(-limx, limx) * z
    ty = uy.clamp(-limy, limy) * z
    zero = torch.zeros_like(z)
    J = torch.stack([torch.stack([cam.fx / z, zero, -cam.fx * tx / (z * z)], -1),
                     torch.stack([zero, cam.fy / z, -cam.fy * ty / (z * z)], -1)], -2)
    T = J @ W
    cov2d = T @ Sigma @ T.transpose(-1, -2) + LOW_PASS * torch.eye(2, dtype=means.dtype, device=means.device)
    A, B, C = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = A * C - B * B
    valid = valid & (det > 0)
    conics = torch.stack([C / det, -B / det, A / det], -1)

    mid = 0.5 * (A + C)
    lam = mid + torch.sqrt(torch.clamp_min(mid * mid - det, 0.1))
    radii = torch.where(valid, torch.ceil(3.0 * torch.sqrt(lam)).to(torch.int64), torch.zeros_like(det, dtype=torch.int64))

    opac = torch.sigmoid(opacity_logits)
    v = means - cam.center
    vnorm = v.norm(dim=-1, keepdim=True)
    dirs = v / vnorm
    k = num_sh_coeffs(sh_degree)
    basis = sh_basis(sh_degree, dirs)
    col_raw = (basis[..., None] * sh[:, :k]).sum(-2) + 0.5
    colors = torch.clamp_min(col_raw, 0.0)

    saved = Saved(qn, qnorm, R, s, M, Sigma, t, tx_clamped, ty_clamped, J, T, cov2d, det, opac, dirs, vnorm,
                  basis, col_raw, valid)
    return (means2d, conics, opac, colors, t[:, 2], radii, valid), saved


# ------------------------------------------------------------------- backward

def preprocess_backward(sv: Saved, cam: Camera, sh_degree: int, n_sh: int,
                        g_means2d, g_conics, g_opac, g_colors):
    """Hand-written chain rule. Inputs are dL/d(outputs); returns dL/d(parameters)."""
    W = cam.R
    valid = sv.valid[:, None]
    g_means2d = torch.where(valid, g_means2d, 0)
    g_conics = torch.where(valid, g_conics, 0)
    x, y, z = sv.t.unbind(-1)
    fx, fy = cam.fx, cam.fy

    # conic = (C, -B, A) / det  ->  dL/dA, dL/dB, dL/dC  (B counted once)
    A, B, C = sv.cov2d[:, 0, 0], sv.cov2d[:, 0, 1], sv.cov2d[:, 1, 1]
    det = sv.det
    ga, gb, gc = g_conics.unbind(-1)
    d2 = det * det
    gA = ga * (-C * C / d2) + gb * (B * C / d2) + gc * (1 / det - A * C / d2)
    gB = ga * (2 * B * C / d2) + gb * (-1 / det - 2 * B * B / d2) + gc * (2 * A * B / d2)
    gC = ga * (1 / det - A * C / d2) + gb * (A * B / d2) + gc * (-A * A / d2)
    # as a symmetric matrix G with L ~ tr(G Sigma'): off-diagonal entries share gB
    G = torch.stack([torch.stack([gA, 0.5 * gB], -1), torch.stack([0.5 * gB, gC], -1)], -2)

    # Sigma' = T Sigma T^T  ->  dL/dSigma = T^T G T,  dL/dT = 2 G T Sigma
    gSigma = sv.T.transpose(-1, -2) @ G @ sv.T
    gT = 2 * G @ sv.T @ sv.Sigma
    gJ = gT @ W.T                                                    # T = J W

    # J entries -> camera-space point t (with the frustum clamp inside J)
    limx, limy = FRUSTUM_SLACK * cam.tan_fovx, FRUSTUM_SLACK * cam.tan_fovy
    tx = torch.where(sv.tx_clamped, torch.sign(x / z) * limx * z, x)
    ty = torch.where(sv.ty_clamped, torch.sign(y / z) * limy * z, y)
    dtx_dx = torch.where(sv.tx_clamped, 0.0, 1.0).to(x.dtype)
    dtx_dz = torch.where(sv.tx_clamped, torch.sign(x / z) * limx, 0.0).to(x.dtype)
    dty_dy = torch.where(sv.ty_clamped, 0.0, 1.0).to(x.dtype)
    dty_dz = torch.where(sv.ty_clamped, torch.sign(y / z) * limy, 0.0).to(x.dtype)
    z2, z3 = z * z, z * z * z
    gJ00, gJ02, gJ11, gJ12 = gJ[:, 0, 0], gJ[:, 0, 2], gJ[:, 1, 1], gJ[:, 1, 2]
    g_tx = gJ02 * (-fx / z2)
    g_ty = gJ12 * (-fy / z2)
    gtx_ = g_tx * dtx_dx
    gty_ = g_ty * dty_dy
    gtz_ = (gJ00 * (-fx / z2) + gJ11 * (-fy / z2) + gJ02 * (2 * fx * tx / z3) + gJ12 * (2 * fy * ty / z3)
            + g_tx * dtx_dz + g_ty * dty_dz)
    # mu2d = (fx x/z + cx, fy y/z + cy)
    gu, gv = g_means2d.unbind(-1)
    gtx_ = gtx_ + gu * fx / z
    gty_ = gty_ + gv * fy / z
    gtz_ = gtz_ - gu * fx * x / z2 - gv * fy * y / z2
    g_t = torch.where(valid, torch.stack([gtx_, gty_, gtz_], -1), 0)
    g_means = g_t @ W                                                # t = W mu + t_cam
    gSigma = torch.where(valid[..., None], gSigma, 0)

    # Sigma = M M^T, M = R diag(s)
    gM = 2 * gSigma @ sv.M
    g_s = (gM * sv.R).sum(-2)                                        # sum_i dL/dM_ij R_ij
    g_log_scales = g_s * sv.s
    gR = gM * sv.s[:, None, :]
    g_qn = rotmat_backward(sv.qn, gR)
    g_quats = (g_qn - (g_qn * sv.qn).sum(-1, keepdim=True) * sv.qn) / sv.qnorm

    # opacity = sigmoid(logit)
    g_logits = g_opac * sv.opac * (1 - sv.opac)

    # colour = max(sum_k Y_k(d) c_k + 0.5, 0)
    g_col = torch.where(sv.col_raw > 0, g_colors, 0)
    k = num_sh_coeffs(sh_degree)
    g_sh = torch.zeros(g_col.shape[0], n_sh, 3, dtype=g_col.dtype, device=g_col.device)
    g_sh[:, :k] = sv.basis[..., None] * g_col[:, None, :]
    # the colour's dependence on the view direction (-> dL/dmu) is in colour_dir_backward
    return g_means, g_log_scales, g_quats, g_logits, g_sh


def colour_dir_backward(sv: Saved, sh: torch.Tensor, sh_degree: int, g_colors: torch.Tensor) -> torch.Tensor:
    """Contribution of the view-dependent colour to dL/dmu."""
    if sh_degree == 0:
        return torch.zeros_like(sv.dirs)
    k = num_sh_coeffs(sh_degree)
    g_col = torch.where(sv.col_raw > 0, g_colors, 0)
    jac = sh_basis_jacobian(sh_degree, sv.dirs)                     # (N, k, 3)
    coef_dot = (sh[:, :k] * g_col[:, None, :]).sum(-1)               # (N, k)
    g_dir = (coef_dot[..., None] * jac).sum(1)                       # (N, 3)
    d = sv.dirs
    return (g_dir - (g_dir * d).sum(-1, keepdim=True) * d) / sv.vnorm


class PreprocessFn(torch.autograd.Function):
    """Autograd wrapper: forward runs `preprocess_forward` without building a graph, backward is
    `preprocess_backward` + `colour_dir_backward` (no autograd involved)."""

    @staticmethod
    def forward(ctx, means, log_scales, quats, opacity_logits, sh_dc, sh_rest, cam, sh_degree):
        sh = torch.cat([sh_dc, sh_rest], 1)
        outs, sv = preprocess_forward(means, log_scales, quats, opacity_logits, sh, cam, sh_degree)
        ctx.sv, ctx.cam, ctx.sh_degree = sv, cam, sh_degree
        ctx.save_for_backward(sh)
        ctx.n_dc = sh_dc.shape[1]
        means2d, conics, opac, colors, depths, radii, valid = outs
        ctx.mark_non_differentiable(depths, radii, valid)
        return means2d, conics, opac, colors, depths, radii, valid

    @staticmethod
    def backward(ctx, g_means2d, g_conics, g_opac, g_colors, _gd, _gr, _gv):
        (sh,) = ctx.saved_tensors
        sv = ctx.sv
        zeros = lambda t: torch.zeros_like(t)
        g_means2d = zeros(sv.t[:, :2]) if g_means2d is None else g_means2d
        g_conics = zeros(sv.t) if g_conics is None else g_conics
        g_opac = zeros(sv.opac) if g_opac is None else g_opac
        g_colors = zeros(sv.t) if g_colors is None else g_colors
        g_means, g_ls, g_q, g_logit, g_sh = preprocess_backward(sv, ctx.cam, ctx.sh_degree, sh.shape[1],
                                                                g_means2d, g_conics, g_opac, g_colors)
        g_means = g_means + colour_dir_backward(sv, sh, ctx.sh_degree, g_colors)
        return g_means, g_ls, g_q, g_logit, g_sh[:, :ctx.n_dc], g_sh[:, ctx.n_dc:], None, None
