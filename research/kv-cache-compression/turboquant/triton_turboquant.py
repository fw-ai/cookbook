"""
Fused TurboQuant attention scoring via Triton.

Two-term unbiased estimator in a single kernel pass:
  score = <q_rot, centroids[idx]>                           ← term 1 (MSE)
        + residual_norm * sqrt(π/2)/m * <q_sketch, signs>   ← term 2 (QJL)

Unlike RotorQuant's kernel, TurboQuant does not normalize keys before
quantizing, so term 1 has no key-norm factor.  The rotation is a global
orthogonal matrix (not per-group Clifford), making the gather-dot flat
across all head_dim dimensions.
"""

import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 32, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 128}, num_warps=4),
    ],
    key=["kv_len", "head_dim"],
)
@triton.jit
def _fused_turboquant_attention_qjl_kernel(
    # MSE inputs
    Q_rot_ptr, K_idx_ptr, C_ptr,
    # QJL inputs
    Q_sketch_ptr, QJL_signs_ptr, Res_norms_ptr,
    # Output
    Out_ptr,
    # Dimensions
    kv_len,
    head_dim: tl.constexpr,
    qjl_m: tl.constexpr,
    n_q_heads, n_kv_heads,
    scale,
    qjl_scale,
    # Strides — Q_rot: [BH_q, head_dim]
    stride_qr_bh, stride_qr_d,
    # Strides — K_idx: [BH_kv, kv_len, head_dim]
    stride_ki_bh, stride_ki_s, stride_ki_d,
    # Strides — Q_sketch: [BH_q, qjl_m]
    stride_qs_bh, stride_qs_d,
    # Strides — QJL_signs: [BH_kv, kv_len, qjl_m]
    stride_js_bh, stride_js_s, stride_js_d,
    # Strides — Res_norms: [BH_kv, kv_len]
    stride_rn_bh, stride_rn_s,
    # Strides — Out: [BH_q, kv_len]
    stride_o_bh, stride_o_s,
    # Block sizes
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Two-term TurboQuant attention: MSE gather-dot + QJL sign correction."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    # GQA mapping
    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    gqa_ratio = n_q_heads // n_kv_heads
    kv_head_idx = q_head_idx // gqa_ratio
    kv_bh = batch_idx * n_kv_heads + kv_head_idx

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < kv_len

    # ── Term 1: MSE gather-dot (flat across head_dim) ──
    acc_mse = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for d_start in range(0, head_dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        q_ptrs = Q_rot_ptr + pid_bh * stride_qr_bh + d_offs * stride_qr_d
        q_vals = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        ki_ptrs = (K_idx_ptr + kv_bh * stride_ki_bh
                   + s_offs[:, None] * stride_ki_s
                   + d_offs[None, :] * stride_ki_d)
        mask_2d = s_mask[:, None] & d_mask[None, :]
        k_idx = tl.load(ki_ptrs, mask=mask_2d, other=0).to(tl.int32)
        k_vals = tl.load(C_ptr + k_idx, mask=mask_2d, other=0.0).to(tl.float32)

        acc_mse += tl.sum(k_vals * q_vals[None, :], axis=1)

    # ── Term 2: QJL correction ──
    acc_qjl = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for d_start in range(0, qjl_m, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask_q = d_offs < qjl_m

        qs_ptrs = Q_sketch_ptr + pid_bh * stride_qs_bh + d_offs * stride_qs_d
        qs_vals = tl.load(qs_ptrs, mask=d_mask_q, other=0.0).to(tl.float32)

        sign_ptrs = (QJL_signs_ptr + kv_bh * stride_js_bh
                     + s_offs[:, None] * stride_js_s
                     + d_offs[None, :] * stride_js_d)
        mask_2d_q = s_mask[:, None] & d_mask_q[None, :]
        signs = tl.load(sign_ptrs, mask=mask_2d_q, other=0).to(tl.float32)

        acc_qjl += tl.sum(qs_vals[None, :] * signs, axis=1)

    # ── Combine: no key-norm factor on term 1 (TurboQuant doesn't normalize) ──
    rn_ptrs = Res_norms_ptr + kv_bh * stride_rn_bh + s_offs * stride_rn_s
    res_norms = tl.load(rn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    term1 = acc_mse
    term2 = res_norms * qjl_scale * acc_qjl
    scores = (term1 + term2) * scale

    o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
    tl.store(o_ptrs, scores, mask=s_mask)


def triton_fused_turboquant_attention_qjl(
    q_rotated: torch.Tensor,      # [batch, n_q_heads, q_len, head_dim]
    q_sketch: torch.Tensor,       # [batch, n_q_heads, q_len, qjl_m]
    key_indices: torch.Tensor,    # [batch, n_kv_heads, kv_len, head_dim] uint8
    qjl_signs: torch.Tensor,     # [batch, n_kv_heads, kv_len, qjl_m] int8/float
    residual_norms: torch.Tensor, # [batch, n_kv_heads, kv_len] fp16/fp32
    centroids: torch.Tensor,     # [n_levels] float32
    scale: float,
    qjl_correction_scale: float,
) -> torch.Tensor:
    """Fused TurboQuant MSE + QJL attention scores via Triton.

    Args:
        q_rotated: Pre-rotated queries (q @ Pi_T).
        q_sketch: Pre-sketched queries (q @ S_T).
        key_indices: Lloyd-Max codebook indices per dim.
        qjl_signs: Sign of projected residual, {+1, -1}.
        residual_norms: L2 norm of quantization residual per key.
        centroids: 1-D Lloyd-Max centroid table.
        scale: Attention scale (typically 1/sqrt(head_dim)).
        qjl_correction_scale: sqrt(pi/2) / qjl_dim.

    Returns:
        Attention scores [batch, n_q_heads, q_len, kv_len].
    """
    batch, n_q_heads, q_len, head_dim = q_rotated.shape
    _, n_kv_heads, kv_len, qjl_m = qjl_signs.shape

    q_flat = q_rotated.reshape(batch * n_q_heads * q_len, head_dim).contiguous().float()
    qs_flat = q_sketch.reshape(batch * n_q_heads * q_len, qjl_m).contiguous().float()
    ki_flat = key_indices.reshape(batch * n_kv_heads, kv_len, head_dim).contiguous()
    sign_flat = qjl_signs.reshape(batch * n_kv_heads, kv_len, qjl_m).contiguous().float()
    rn_flat = residual_norms.reshape(batch * n_kv_heads, kv_len).contiguous().float()
    centroids = centroids.contiguous().float()

    effective_q_heads = n_q_heads * q_len
    out = torch.empty(batch * effective_q_heads, kv_len,
                      device=q_rotated.device, dtype=torch.float32)

    grid = lambda meta: (batch * effective_q_heads,
                         triton.cdiv(kv_len, meta["BLOCK_S"]))

    _fused_turboquant_attention_qjl_kernel[grid](
        q_flat, ki_flat, centroids,
        qs_flat, sign_flat, rn_flat,
        out,
        kv_len, head_dim, qjl_m,
        effective_q_heads, n_kv_heads,
        scale, qjl_correction_scale,
        q_flat.stride(0), q_flat.stride(1),
        ki_flat.stride(0), ki_flat.stride(1), ki_flat.stride(2),
        qs_flat.stride(0), qs_flat.stride(1),
        sign_flat.stride(0), sign_flat.stride(1), sign_flat.stride(2),
        rn_flat.stride(0), rn_flat.stride(1),
        out.stride(0), out.stride(1),
    )

    return out.reshape(batch, n_q_heads, q_len, kv_len)


def pre_rotate_query_turbo(query: torch.Tensor, Pi_T: torch.Tensor) -> torch.Tensor:
    """Pre-rotate queries through the global orthogonal matrix: q_rot = q @ Pi_T.

    Args:
        query: [batch, n_q_heads, q_len, head_dim]
        Pi_T: [head_dim, head_dim] — transposed rotation matrix.
    """
    B, H, Q, D = query.shape
    return (query.float().reshape(-1, D) @ Pi_T).reshape(B, H, Q, D)


def pre_sketch_query_turbo(query: torch.Tensor, S_T: torch.Tensor) -> torch.Tensor:
    """Pre-project queries through QJL matrix: q_sketch = q @ S_T.

    Args:
        query: [batch, n_q_heads, q_len, head_dim]
        S_T: [head_dim, qjl_dim] — transposed QJL projection.
    """
    B, H, Q, D = query.shape
    qjl_m = S_T.shape[1]
    return (query.float().reshape(-1, D) @ S_T).reshape(B, H, Q, qjl_m)
