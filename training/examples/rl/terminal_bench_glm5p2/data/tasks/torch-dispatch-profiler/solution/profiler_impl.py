"""
Performance profiler using TorchDispatchMode.

Implements:
1. FLOP counting with per-module attribution using TorchDispatchMode
   to intercept aten matmul operations during forward pass
2. Backward FLOP estimation based on autograd structure
3. Matmul shape analysis for cache alignment, wave quantization, and roofline
"""

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
import math
from collections import defaultdict


# ──────────────────────────────────────────────
# FLOP formulas for matmul-family ops
# ──────────────────────────────────────────────

def _mm_flops(args):
    a, b = args[0], args[1]
    M, K = a.shape
    _, N = b.shape
    return 2 * M * K * N


def _addmm_flops(args):
    # addmm(bias, mat1, mat2) -> bias + mat1 @ mat2
    a, b = args[1], args[2]
    M, K = a.shape
    _, N = b.shape
    return 2 * M * K * N


def _bmm_flops(args):
    a, b = args[0], args[1]
    B, M, K = a.shape
    _, _, N = b.shape
    return 2 * B * M * K * N


_FLOP_MAP = {}


def _ensure_flop_map():
    global _FLOP_MAP
    if _FLOP_MAP:
        return
    _FLOP_MAP[torch.ops.aten.mm] = _mm_flops
    _FLOP_MAP[torch.ops.aten.addmm] = _addmm_flops
    _FLOP_MAP[torch.ops.aten.bmm] = _bmm_flops


# ──────────────────────────────────────────────
# TorchDispatchMode-based FLOP counter
# ──────────────────────────────────────────────

class _FlopCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        _ensure_flop_map()
        self.module_flops = defaultdict(int)
        self._module_stack = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)

        packet = func.overloadpacket if hasattr(func, "overloadpacket") else func
        if packet in _FLOP_MAP:
            flops = _FLOP_MAP[packet](args)
            if self._module_stack:
                name = self._module_stack[-1]
                self.module_flops[name] += flops

        return result


# ──────────────────────────────────────────────
# profile_model
# ──────────────────────────────────────────────

def profile_model(model, sample_input):
    """Profile per-module FLOPs for forward and backward passes.

    Uses TorchDispatchMode to intercept aten matmul ops (mm, addmm, bmm)
    during the forward pass and attribute them to modules via forward hooks.

    Backward FLOPs are computed analytically from the forward FLOPs:
    - Each linear layer's backward always computes grad_weight (1x fwd FLOPs)
    - It also computes grad_input (another 1x fwd FLOPs) when the layer's
      input requires grad
    - For a chain of parameterized layers, every layer after the first always
      receives input that requires grad
    - The first layer's input requires grad only if sample_input.requires_grad
    """
    counter = _FlopCounter()
    handles = []
    execution_order = []

    # Identify first-level child modules
    child_name_map = {}
    for name, child in model.named_modules():
        if name and "." not in name:
            child_name_map[child] = name

    # Forward hooks to track module context and execution order
    def _make_fwd_pre(n):
        def hook(mod, inp):
            if n not in execution_order:
                execution_order.append(n)
            counter._module_stack.append(n)
        return hook

    def _make_fwd_post(n):
        def hook(mod, inp, output):
            if counter._module_stack and counter._module_stack[-1] == n:
                counter._module_stack.pop()
        return hook

    for child, name in child_name_map.items():
        handles.append(child.register_forward_pre_hook(_make_fwd_pre(name)))
        handles.append(child.register_forward_hook(_make_fwd_post(name)))

    # Run forward under TorchDispatchMode to count forward FLOPs
    with counter:
        model(sample_input)

    for h in handles:
        h.remove()

    # Collect forward FLOPs per module
    forward_flops = {}
    total_fwd = 0
    for name in execution_order:
        fwd = counter.module_flops.get(name, 0)
        forward_flops[name] = fwd
        total_fwd += fwd

    # Compute backward FLOPs analytically
    input_requires_grad = sample_input.requires_grad

    modules = {}
    total_bwd = 0
    for i, name in enumerate(execution_order):
        fwd = forward_flops[name]
        if i == 0 and not input_requires_grad:
            bwd = fwd       # only grad_weight
        else:
            bwd = 2 * fwd   # grad_weight + grad_input
        modules[name] = {
            "forward_flops": fwd,
            "backward_flops": bwd,
        }
        total_bwd += bwd

    return {
        "modules": modules,
        "total_forward_flops": total_fwd,
        "total_backward_flops": total_bwd,
    }


# ──────────────────────────────────────────────
# analyze_matmul
# ──────────────────────────────────────────────

def analyze_matmul(M, K, N, dtype_bytes, hardware):
    """Analyze a [M,K]@[K,N] matmul for cache alignment, wave quantization,
    and roofline classification."""
    flops = 2 * M * K * N
    memory_bytes = (M * K + K * N + M * N) * dtype_bytes
    arithmetic_intensity = flops / memory_bytes

    ridge_point = hardware["peak_flops"] / hardware["peak_bandwidth_bytes_per_sec"]
    is_compute_bound = arithmetic_intensity > ridge_point

    cache_line_elems = hardware["cache_line_bytes"] // dtype_bytes
    k_aligned = (K % cache_line_elems) == 0
    n_aligned = (N % cache_line_elems) == 0

    def _pad_to_multiple(val, align):
        if val % align == 0:
            return val
        return ((val // align) + 1) * align

    k_pad_to = _pad_to_multiple(K, cache_line_elems)
    n_pad_to = _pad_to_multiple(N, cache_line_elems)

    tile_m = hardware["tile_m"]
    tile_n = hardware["tile_n"]
    tile_grid = math.ceil(M / tile_m) * math.ceil(N / tile_n)
    sm_count = hardware["sm_count"]
    wave_count = math.ceil(tile_grid / sm_count)
    wave_efficiency = tile_grid / (wave_count * sm_count)

    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "is_compute_bound": is_compute_bound,
        "k_aligned": k_aligned,
        "n_aligned": n_aligned,
        "k_pad_to": k_pad_to,
        "n_pad_to": n_pad_to,
        "tile_grid": tile_grid,
        "wave_count": wave_count,
        "wave_efficiency": wave_efficiency,
    }
