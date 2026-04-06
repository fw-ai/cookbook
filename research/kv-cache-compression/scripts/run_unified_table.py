#!/usr/bin/env python3
"""Unified cross-method kernel table using retained RotorQuant + TurboQuant code.

Differences from the original ``run_kernel_table.py`` (phase-1 harness):

- RotorQuant uses the **trimmed layout** (d=128 vector-grade indices via
  ``build_trimmed_mse_quantization``), matching TurboQuant's effective
  bits/dim exactly (nominal + 0.125).
- TurboQuant is measured in **both cold and steady-state** timing families,
  with both the PyTorch matmul path and the fused Triton kernel.
- Both methods report ``pre_rot_ms``, ``pre_sk_ms``, ``e2e_est_ms`` for
  their Triton score paths.
- GPU selection via ``--gpu`` flag (sets ``CUDA_VISIBLE_DEVICES``).
- ``--methods`` flag to run a subset of methods (e.g., ``--methods turboquant``).
- Default ``--repo-root`` points at this package root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_REPO_ROOT = PROJECT_ROOT
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "kernel_table_phase1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "unified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified cross-method kernel table.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gpu", type=int, default=None,
                        help="Physical GPU index (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--context", action="append", type=int,
                        help="Context length override (can repeat).")
    parser.add_argument("--warmup", type=int, help="Override warmup iterations.")
    parser.add_argument("--iterations", type=int, help="Override timed iterations.")
    parser.add_argument("--seed", type=int, help="Override workload seed.")
    parser.add_argument("--methods", nargs="+",
                        choices=["conventional", "turboquant", "rotorquant"],
                        help="Run only these methods (default: all).")
    return parser.parse_args()


def load_config(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def install_repo_path(repo_root: pathlib.Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_heads, seq_len, dim = hidden_states.shape
    return (hidden_states[:, :, None, :, :]
            .expand(bsz, n_heads, n_rep, seq_len, dim)
            .reshape(bsz, n_heads * n_rep, seq_len, dim))


def normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def time_cuda_fn(fn, n_warmup: int, n_iter: int) -> tuple[float, Any]:
    last_result = None
    for _ in range(n_warmup):
        last_result = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        last_result = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n_iter * 1000.0
    return elapsed_ms, last_result


# ── Conventional ──

def build_conventional_state(keys: torch.Tensor, bits: int) -> dict[str, Any]:
    if bits == 16:
        return {"bits": bits, "keys_bf16": keys.to(torch.bfloat16)}
    levels = 2 ** bits
    step = 2.0 / (levels - 1)
    idx = torch.round((keys.float() + 1.0) / step).clamp_(0, levels - 1).to(torch.uint8)
    return {"bits": bits, "indices": idx, "step": step}


def score_conventional(query, state, attn_scale, n_kv_heads):
    if state["bits"] == 16:
        keys = state["keys_bf16"]
    else:
        keys = state["indices"].float() * state["step"] - 1.0
        keys = keys.to(query.dtype)
    n_rep = query.shape[1] // n_kv_heads
    full_keys = repeat_kv(keys, n_rep)
    return torch.matmul(query, full_keys.transpose(-2, -1)) * attn_scale


# ── TurboQuant ──

def build_turboquant_cold(keys, bits, seed):
    from turboquant.turboquant import TurboQuantProd
    batch_size, n_kv_heads, _, head_dim = keys.shape
    quantizer = TurboQuantProd(head_dim, bits, seed=seed, device=keys.device)
    compressed = []
    for b in range(batch_size):
        for h in range(n_kv_heads):
            compressed.append(quantizer.quantize(keys[b, h].float()))
    return {"bits": bits, "quantizer": quantizer, "compressed": compressed,
            "n_kv_heads": n_kv_heads, "head_dim": head_dim}


def _quantize_with(q_fn, quantizer, keys):
    """Re-quantize keys using q_fn (steady-state helper)."""
    batch_size, n_kv_heads = keys.shape[:2]
    compressed = []
    for b in range(batch_size):
        for h in range(n_kv_heads):
            compressed.append(q_fn(keys[b, h].float()))
    return {"bits": quantizer.bits, "quantizer": quantizer,
            "compressed": compressed, "n_kv_heads": n_kv_heads,
            "head_dim": quantizer.d}


def score_turboquant(query, state, attn_scale):
    batch_size, n_q_heads, q_len, head_dim = query.shape
    quantizer = state["quantizer"]
    n_kv_heads = state["n_kv_heads"]
    gqa_ratio = n_q_heads // n_kv_heads
    kv_len = state["compressed"][0]["mse_indices"].shape[0]
    scores = torch.empty(batch_size, n_q_heads, q_len, kv_len,
                         device=query.device, dtype=torch.float32)
    for b in range(batch_size):
        for qh in range(n_q_heads):
            kvh = qh // gqa_ratio
            comp = state["compressed"][b * n_kv_heads + kvh]
            q_vec = query[b, qh, 0].float()
            scores[b, qh, 0] = quantizer.inner_product(q_vec, comp) * attn_scale
    return scores


# ── TurboQuant Triton scoring ──

def _stack_turbo_compressed(state):
    """Reshape per-head compressed list into batch tensors for the Triton kernel."""
    compressed_list = state["compressed"]
    quantizer = state["quantizer"]
    n_kv_heads = state["n_kv_heads"]
    head_dim = state["head_dim"]
    n_total = len(compressed_list)
    batch_size = n_total // n_kv_heads
    kv_len = compressed_list[0]["mse_indices"].shape[0]

    all_idx = torch.stack([c["mse_indices"] for c in compressed_list])
    all_signs = torch.stack([c["qjl_signs"] for c in compressed_list])
    all_rnorms = torch.stack([c["residual_norm"] for c in compressed_list])

    return {
        "key_indices": all_idx.reshape(batch_size, n_kv_heads, kv_len, head_dim).contiguous(),
        "qjl_signs": all_signs.reshape(batch_size, n_kv_heads, kv_len, head_dim).contiguous(),
        "residual_norms": all_rnorms.reshape(batch_size, n_kv_heads, kv_len).contiguous(),
        "centroids": quantizer.mse.centroids,
        "Pi_T": quantizer.mse.Pi_T,
        "S_T": quantizer.S_T,
        "qjl_correction_scale": quantizer._qjl_correction_scale,
    }


def score_turboquant_triton(query, triton_state, attn_scale):
    from turboquant.triton_turboquant import (
        triton_fused_turboquant_attention_qjl,
        pre_rotate_query_turbo,
        pre_sketch_query_turbo,
    )
    q_rot = pre_rotate_query_turbo(query, triton_state["Pi_T"])
    q_sketch = pre_sketch_query_turbo(query, triton_state["S_T"])
    return triton_fused_turboquant_attention_qjl(
        q_rot, q_sketch,
        triton_state["key_indices"], triton_state["qjl_signs"],
        triton_state["residual_norms"], triton_state["centroids"],
        attn_scale, triton_state["qjl_correction_scale"],
    )


def time_turboquant_triton_stages(query, triton_state, attn_scale, n_warmup, n_iter):
    """Time pre_rotate, pre_sketch, and triton score separately for TurboQuant."""
    from turboquant.triton_turboquant import (
        triton_fused_turboquant_attention_qjl,
        pre_rotate_query_turbo,
        pre_sketch_query_turbo,
    )
    pre_rot_ms, q_rot = time_cuda_fn(
        lambda: pre_rotate_query_turbo(query, triton_state["Pi_T"]),
        n_warmup, n_iter)

    pre_sk_ms, q_sketch = time_cuda_fn(
        lambda: pre_sketch_query_turbo(query, triton_state["S_T"]),
        n_warmup, n_iter)

    triton_ms, _ = time_cuda_fn(
        lambda: triton_fused_turboquant_attention_qjl(
            q_rot, q_sketch,
            triton_state["key_indices"], triton_state["qjl_signs"],
            triton_state["residual_norms"], triton_state["centroids"],
            attn_scale, triton_state["qjl_correction_scale"]),
        n_warmup, n_iter)

    return pre_rot_ms, pre_sk_ms, triton_ms


# ── RotorQuant (trimmed layout) ──

def build_rotorquant_state(keys, bits, seed):
    from turboquant.fused_attention import (
        pre_rotate_query,
        pre_sketch_query,
        triton_fused_attention_qjl,
    )
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import (
        pack_rotors_for_triton,
        triton_rotor_fused_build,
    )
    del pre_rotate_query, pre_sketch_query, triton_fused_attention_qjl

    batch_size, n_kv_heads, kv_len, head_dim = keys.shape
    mse_bits = max(bits - 1, 1)
    rq = RotorQuantMSE(head_dim, mse_bits, seed=seed, device=keys.device)
    packed_rotors = pack_rotors_for_triton(rq.rotors).to(keys.device)
    centroids = getattr(rq, "centroids_vector").to(keys.device)
    bounds = ((centroids[:-1] + centroids[1:]) * 0.5).contiguous()

    flat = keys.reshape(-1, head_dim).float().contiguous()
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    flat_unit = flat / norms

    idx_flat, k_mse_unit = triton_rotor_fused_build(
        flat_unit, packed_rotors, centroids, bounds)
    k_mse = k_mse_unit * norms

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1)
    S = torch.randn(head_dim, head_dim, generator=gen).to(keys.device)

    residual = flat - k_mse
    residual_norms = residual.norm(dim=-1)
    qjl_signs = torch.sign(residual @ S.T).to(torch.int8)
    qjl_signs[qjl_signs == 0] = 1

    return {
        "bits": bits,
        "packed_rotors": packed_rotors,
        "centroids": centroids,
        "centroid_bounds": bounds,
        "S": S,
        "key_indices": idx_flat.reshape(batch_size, n_kv_heads, kv_len, head_dim).contiguous(),
        "key_norms": norms.squeeze(-1).half().reshape(batch_size, n_kv_heads, kv_len).contiguous(),
        "qjl_signs": qjl_signs.reshape(batch_size, n_kv_heads, kv_len, head_dim).contiguous(),
        "residual_norms": residual_norms.half().reshape(batch_size, n_kv_heads, kv_len).contiguous(),
        "head_dim": head_dim,
        "n_groups": rq.n_groups,
    }


def _rotorquant_compress_keys(keys, packed_rotors, centroids, centroid_bounds, S, head_dim):
    """Per-token RotorQuant build work (steady-state: quantizer already exists)."""
    from turboquant.triton_kernels import triton_rotor_fused_build

    batch_size, n_kv_heads, kv_len, _ = keys.shape
    flat = keys.reshape(-1, head_dim).float().contiguous()
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    flat_unit = flat / norms

    idx_flat, k_mse_unit = triton_rotor_fused_build(
        flat_unit, packed_rotors, centroids, centroid_bounds)
    k_mse = k_mse_unit * norms

    residual = flat - k_mse
    residual_norms = residual.norm(dim=-1)
    qjl_signs = torch.sign(residual @ S.T).to(torch.int8)
    qjl_signs[qjl_signs == 0] = 1

    return {
        "key_indices": idx_flat.reshape(batch_size, n_kv_heads, kv_len, head_dim).contiguous(),
        "key_norms": norms.squeeze(-1).half().reshape(batch_size, n_kv_heads, kv_len).contiguous(),
        "qjl_signs": qjl_signs.reshape(batch_size, n_kv_heads, kv_len, head_dim).contiguous(),
        "residual_norms": residual_norms.half().reshape(batch_size, n_kv_heads, kv_len).contiguous(),
    }


def score_rotorquant(query, state, attn_scale):
    from turboquant.fused_attention import (
        pre_rotate_query,
        pre_sketch_query,
        triton_fused_attention_qjl,
    )
    q_rotated = pre_rotate_query(query.float(), state["packed_rotors"], state["head_dim"])
    q_sketch = pre_sketch_query(query.float(), state["S"])
    return triton_fused_attention_qjl(
        q_rotated, q_sketch,
        state["key_indices"], state["key_norms"],
        state["qjl_signs"], state["residual_norms"],
        state["centroids"], attn_scale,
    )


def time_rotorquant_stages(query, state, attn_scale, n_warmup, n_iter):
    """Time pre_rotate, pre_sketch, and triton score separately."""
    from turboquant.fused_attention import (
        pre_rotate_query,
        pre_sketch_query,
        triton_fused_attention_qjl,
    )
    pre_rot_ms, q_rotated = time_cuda_fn(
        lambda: pre_rotate_query(query.float(), state["packed_rotors"], state["head_dim"]),
        n_warmup, n_iter)

    pre_sk_ms, q_sketch = time_cuda_fn(
        lambda: pre_sketch_query(query.float(), state["S"]),
        n_warmup, n_iter)

    triton_ms, _ = time_cuda_fn(
        lambda: triton_fused_attention_qjl(
            q_rotated, q_sketch,
            state["key_indices"], state["key_norms"],
            state["qjl_signs"], state["residual_norms"],
            state["centroids"], attn_scale),
        n_warmup, n_iter)

    return pre_rot_ms, pre_sk_ms, triton_ms


# ── Metrics ──

def metrics_against_reference(reference, candidate):
    ref = reference.float().squeeze(-2)
    cand = candidate.float().squeeze(-2)
    rmse = torch.sqrt(torch.mean((cand - ref) ** 2)).item()
    cos = F.cosine_similarity(
        ref.reshape(-1, ref.shape[-1]),
        cand.reshape(-1, cand.shape[-1]), dim=-1).mean().item()
    top1 = (ref.argmax(dim=-1) == cand.argmax(dim=-1)).float().mean().item() * 100.0
    k = min(5, ref.shape[-1])
    ref_top1 = ref.argmax(dim=-1, keepdim=True)
    cand_topk = cand.topk(k, dim=-1).indices
    top5 = (cand_topk == ref_top1).any(dim=-1).float().mean().item() * 100.0
    return {"score_rmse": rmse, "score_cosine": cos,
            "top1_match_pct": top1, "top5_match_pct": top5}


# ── Main loop ──

def run_context(context, workload, timing_cfg, methods=None):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[workload["dtype"]]
    batch_size = workload["batch_size"]
    n_q_heads = workload["n_query_heads"]
    n_kv_heads = workload["n_kv_heads"]
    q_len = workload["q_len"]
    head_dim = workload["head_dim"]
    seed = workload["seed"] + context
    warmup = timing_cfg["warmup"]
    iterations = timing_cfg["iterations"]
    attn_scale = 1.0 / math.sqrt(head_dim)
    device = torch.device("cuda")
    run_all = methods is None
    methods = set(methods) if methods else set()

    torch.manual_seed(seed)
    keys = normalize_last_dim(torch.randn(batch_size, n_kv_heads, context, head_dim,
                                          device=device, dtype=dtype))
    query = normalize_last_dim(torch.randn(batch_size, n_q_heads, q_len, head_dim,
                                           device=device, dtype=dtype))

    ref_state = build_conventional_state(keys, bits=16)
    ref_score_ms, reference_scores = time_cuda_fn(
        lambda: score_conventional(query, ref_state, attn_scale, n_kv_heads),
        n_warmup=warmup, n_iter=iterations)

    rows: list[dict[str, Any]] = []

    # ── Conventional ──
    if run_all or "conventional" in methods:
        for bits in [16, 8, 4, 3, 2]:
            if bits == 16:
                build_ms, score_ms, candidate = 0.0, ref_score_ms, reference_scores
            else:
                state = build_conventional_state(keys, bits)
                build_ms = time_cuda_fn(lambda: build_conventional_state(keys, bits),
                                        warmup, iterations)[0]
                score_ms, candidate = time_cuda_fn(
                    lambda: score_conventional(query, state, attn_scale, n_kv_heads),
                    warmup, iterations)
            row = {
                "context": context, "method": "conventional", "nominal_bits": bits,
                "eff_bits_per_dim": float(bits),
                "timing_family": "cold",
                "build_ms": round(build_ms, 3), "score_ms": round(score_ms, 3),
                "total_ms": round(build_ms + score_ms, 3),
            }
            row.update(metrics_against_reference(reference_scores, candidate))
            rows.append(row)

    # ── TurboQuant cold (PyTorch matmul score) ──
    if run_all or "turboquant" in methods:
        for bits in [8, 4, 3, 2]:
            state = build_turboquant_cold(keys, bits, seed)
            build_ms = time_cuda_fn(lambda: build_turboquant_cold(keys, bits, seed),
                                    warmup, iterations)[0]
            score_ms, candidate = time_cuda_fn(
                lambda: score_turboquant(query, state, attn_scale),
                warmup, iterations)
            eff = bits + 16.0 / head_dim
            row = {
                "context": context, "method": "turboquant", "nominal_bits": bits,
                "eff_bits_per_dim": round(eff, 3),
                "timing_family": "cold",
                "build_ms": round(build_ms, 3), "score_ms": round(score_ms, 3),
                "total_ms": round(build_ms + score_ms, 3),
            }
            row.update(metrics_against_reference(reference_scores, candidate))
            rows.append(row)

    # ── TurboQuant steady-state (cached quantizer + torch.compile, PyTorch score) ──
    if run_all or "turboquant" in methods:
        for bits in [8, 4, 3, 2]:
            from turboquant.turboquant import TurboQuantProd
            quantizer = TurboQuantProd(head_dim, bits, seed=seed, device=keys.device)
            compiled_quantize = torch.compile(quantizer.quantize)

            for _ in range(3):
                _quantize_with(compiled_quantize, quantizer, keys)
            torch.cuda.synchronize()

            build_ms = time_cuda_fn(
                lambda: _quantize_with(compiled_quantize, quantizer, keys),
                warmup, iterations)[0]

            state = _quantize_with(compiled_quantize, quantizer, keys)
            score_ms, candidate = time_cuda_fn(
                lambda: score_turboquant(query, state, attn_scale),
                warmup, iterations)
            eff = bits + 16.0 / head_dim
            row = {
                "context": context, "method": "turboquant", "nominal_bits": bits,
                "eff_bits_per_dim": round(eff, 3),
                "timing_family": "steady_state_cached_compile",
                "build_ms": round(build_ms, 3), "score_ms": round(score_ms, 3),
                "total_ms": round(build_ms + score_ms, 3),
            }
            row.update(metrics_against_reference(reference_scores, candidate))
            rows.append(row)

    # ── TurboQuant cold build + Triton score ──
    if run_all or "turboquant" in methods:
        for bits in [8, 4, 3, 2]:
            state = build_turboquant_cold(keys, bits, seed)
            build_ms = time_cuda_fn(lambda: build_turboquant_cold(keys, bits, seed),
                                    warmup, iterations)[0]

            triton_state = _stack_turbo_compressed(state)
            pre_rot_ms, pre_sk_ms, triton_score_ms = time_turboquant_triton_stages(
                query, triton_state, attn_scale, warmup, iterations)
            e2e_est_ms = pre_rot_ms + pre_sk_ms + triton_score_ms

            _, candidate = time_cuda_fn(
                lambda: score_turboquant_triton(query, triton_state, attn_scale),
                warmup, iterations)

            eff = bits + 16.0 / head_dim
            row = {
                "context": context, "method": "turboquant", "nominal_bits": bits,
                "eff_bits_per_dim": round(eff, 3),
                "timing_family": "cold_build_triton_score",
                "build_ms": round(build_ms, 3),
                "score_ms": round(triton_score_ms, 3),
                "pre_rot_ms": round(pre_rot_ms, 3),
                "pre_sk_ms": round(pre_sk_ms, 3),
                "e2e_est_ms": round(e2e_est_ms, 3),
                "total_ms": round(build_ms + e2e_est_ms, 3),
            }
            row.update(metrics_against_reference(reference_scores, candidate))
            rows.append(row)

    # ── TurboQuant steady-state + Triton score ──
    if run_all or "turboquant" in methods:
        for bits in [8, 4, 3, 2]:
            from turboquant.turboquant import TurboQuantProd
            quantizer = TurboQuantProd(head_dim, bits, seed=seed, device=keys.device)
            compiled_quantize = torch.compile(quantizer.quantize)

            for _ in range(3):
                _quantize_with(compiled_quantize, quantizer, keys)
            torch.cuda.synchronize()

            build_ms = time_cuda_fn(
                lambda: _quantize_with(compiled_quantize, quantizer, keys),
                warmup, iterations)[0]

            state = _quantize_with(compiled_quantize, quantizer, keys)
            triton_state = _stack_turbo_compressed(state)

            pre_rot_ms, pre_sk_ms, triton_score_ms = time_turboquant_triton_stages(
                query, triton_state, attn_scale, warmup, iterations)
            e2e_est_ms = pre_rot_ms + pre_sk_ms + triton_score_ms

            _, candidate = time_cuda_fn(
                lambda: score_turboquant_triton(query, triton_state, attn_scale),
                warmup, iterations)

            eff = bits + 16.0 / head_dim
            row = {
                "context": context, "method": "turboquant", "nominal_bits": bits,
                "eff_bits_per_dim": round(eff, 3),
                "timing_family": "steady_state_triton_score",
                "build_ms": round(build_ms, 3),
                "score_ms": round(triton_score_ms, 3),
                "pre_rot_ms": round(pre_rot_ms, 3),
                "pre_sk_ms": round(pre_sk_ms, 3),
                "e2e_est_ms": round(e2e_est_ms, 3),
                "total_ms": round(build_ms + e2e_est_ms, 3),
            }
            row.update(metrics_against_reference(reference_scores, candidate))
            rows.append(row)

    # ── RotorQuant cold (trimmed layout) ──
    if run_all or "rotorquant" in methods:
        for bits in [8, 4, 3, 2]:
            state = build_rotorquant_state(keys, bits, seed)
            build_ms = time_cuda_fn(lambda: build_rotorquant_state(keys, bits, seed),
                                    warmup, iterations)[0]

            pre_rot_ms, pre_sk_ms, triton_score_ms = time_rotorquant_stages(
                query, state, attn_scale, warmup, iterations)
            e2e_est_ms = pre_rot_ms + pre_sk_ms + triton_score_ms

            _, candidate = time_cuda_fn(
                lambda: score_rotorquant(query, state, attn_scale),
                warmup, iterations)

            eff = bits + 16.0 / head_dim
            row = {
                "context": context, "method": "rotorquant", "nominal_bits": bits,
                "eff_bits_per_dim": round(eff, 3),
                "timing_family": "cold_build_triton_score",
                "build_ms": round(build_ms, 3),
                "score_ms": round(triton_score_ms, 3),
                "pre_rot_ms": round(pre_rot_ms, 3),
                "pre_sk_ms": round(pre_sk_ms, 3),
                "e2e_est_ms": round(e2e_est_ms, 3),
                "total_ms": round(build_ms + e2e_est_ms, 3),
            }
            row.update(metrics_against_reference(reference_scores, candidate))
            rows.append(row)

    # ── RotorQuant steady-state (pre-built quantizer, only per-key work timed) ──
    if run_all or "rotorquant" in methods:
        for bits in [8, 4, 3, 2]:
            from turboquant.rotorquant import RotorQuantMSE
            from turboquant.triton_kernels import pack_rotors_for_triton

            mse_bits = max(bits - 1, 1)
            rq = RotorQuantMSE(head_dim, mse_bits, seed=seed, device=keys.device)
            packed_rotors = pack_rotors_for_triton(rq.rotors).to(keys.device)
            centroids = getattr(rq, "centroids_vector").to(keys.device)
            bounds = ((centroids[:-1] + centroids[1:]) * 0.5).contiguous()
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed + 1)
            S = torch.randn(head_dim, head_dim, generator=gen).to(keys.device)

            for _ in range(3):
                _rotorquant_compress_keys(keys, packed_rotors, centroids, bounds, S, head_dim)
            torch.cuda.synchronize()

            build_ms = time_cuda_fn(
                lambda: _rotorquant_compress_keys(keys, packed_rotors, centroids, bounds, S, head_dim),
                warmup, iterations)[0]

            compressed = _rotorquant_compress_keys(keys, packed_rotors, centroids, bounds, S, head_dim)
            state = {
                "bits": bits, "packed_rotors": packed_rotors, "centroids": centroids,
                "centroid_bounds": bounds, "S": S, "head_dim": head_dim, "n_groups": rq.n_groups,
                **compressed,
            }

            pre_rot_ms, pre_sk_ms, triton_score_ms = time_rotorquant_stages(
                query, state, attn_scale, warmup, iterations)
            e2e_est_ms = pre_rot_ms + pre_sk_ms + triton_score_ms

            _, candidate = time_cuda_fn(
                lambda: score_rotorquant(query, state, attn_scale),
                warmup, iterations)

            eff = bits + 16.0 / head_dim
            row = {
                "context": context, "method": "rotorquant", "nominal_bits": bits,
                "eff_bits_per_dim": round(eff, 3),
                "timing_family": "steady_state_triton_score",
                "build_ms": round(build_ms, 3),
                "score_ms": round(triton_score_ms, 3),
                "pre_rot_ms": round(pre_rot_ms, 3),
                "pre_sk_ms": round(pre_sk_ms, 3),
                "e2e_est_ms": round(e2e_est_ms, 3),
                "total_ms": round(build_ms + e2e_est_ms, 3),
            }
            row.update(metrics_against_reference(reference_scores, candidate))
            rows.append(row)

    return rows


def write_csv(rows, path):
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    lines = ["# Unified Cross-Method Kernel Table", "",
             "Generated by `scripts/run_unified_table.py` using merged retained code.",
             f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    for context in sorted({r["context"] for r in rows}):
        lines.append(f"## Context {context}")
        lines.append("")
        lines.append("| method | nominal bits | eff bits/dim | timing family | "
                      "score cos | top-1 % | score ms | build ms | total ms | "
                      "pre_rot ms | pre_sk ms | e2e est ms |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ctx_rows = [r for r in rows if r["context"] == context]
        method_order = {"conventional": 0, "turboquant": 1, "rotorquant": 2}
        family_order = {
            "cold": 0,
            "steady_state_cached_compile": 1,
            "cold_build_triton_score": 2,
            "steady_state_triton_score": 3,
        }
        ctx_rows.sort(key=lambda r: (method_order.get(r["method"], 9),
                                     family_order.get(r["timing_family"], 9),
                                     -r["nominal_bits"]))
        for r in ctx_rows:
            fmt = lambda v, d=3: f"{v:.{d}f}" if isinstance(v, float) else (str(v) if v is not None else "")
            lines.append(
                f"| {r['method']} | {r['nominal_bits']} | {fmt(r['eff_bits_per_dim'])} | "
                f"{r['timing_family']} | "
                f"{fmt(r.get('score_cosine', ''), 4)} | {fmt(r.get('top1_match_pct', ''), 1)} | "
                f"{fmt(r.get('score_ms', ''))} | {fmt(r.get('build_ms', ''))} | "
                f"{fmt(r.get('total_ms', ''))} | "
                f"{fmt(r.get('pre_rot_ms', ''))} | {fmt(r.get('pre_sk_ms', ''))} | "
                f"{fmt(r.get('e2e_est_ms', ''))} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    repo_root = pathlib.Path(args.repo_root).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    config_path = pathlib.Path(args.config).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    install_repo_path(repo_root)

    if args.context:
        config["matrix"]["contexts"] = args.context
    if args.warmup is not None:
        config["timing"]["warmup"] = args.warmup
    if args.iterations is not None:
        config["timing"]["iterations"] = args.iterations
    if args.seed is not None:
        config["workload"]["seed"] = args.seed

    all_rows: list[dict[str, Any]] = []
    for context in config["matrix"]["contexts"]:
        print(f"Running context={context} ...")
        all_rows.extend(run_context(context, config["workload"], config["timing"],
                                    methods=args.methods))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"unified_table-{stamp}.csv"
    md_path = output_dir / f"unified_table-{stamp}.md"

    write_csv(all_rows, csv_path)
    write_markdown(all_rows, md_path)

    print(f"csv={csv_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
