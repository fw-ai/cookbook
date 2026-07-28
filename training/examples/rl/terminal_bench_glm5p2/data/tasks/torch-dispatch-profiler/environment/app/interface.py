"""
Interface specification for the profiler.

Implement these two functions in /app/profiler.py.

1. profile_model(model: nn.Module, sample_input: torch.Tensor) -> dict

   Run the model forward (and backward via .sum().backward()), returning
   per-module FLOP counts with correct module attribution in both passes.

   The sample_input may or may not have requires_grad=True. If it does NOT
   require grad, the first layer's backward should only count grad_weight
   (not grad_input), resulting in 1x forward FLOPs instead of 2x.

   Return format:
   {
       "modules": {
           "fc1": {"forward_flops": int, "backward_flops": int},
           "fc2": {"forward_flops": int, "backward_flops": int},
           ...
       },
       "total_forward_flops": int,
       "total_backward_flops": int,
   }

   FLOP convention: FLOPs = 2 * M * K * N for a [M,K] @ [K,N] matmul.

   Module names should be the attribute names as registered in the model
   (e.g., "fc1", "fc2", "fc3" for TargetModel).

2. analyze_matmul(M: int, K: int, N: int, dtype_bytes: int, hardware: dict) -> dict

   Analyze a single matmul shape [M,K] @ [K,N] for performance characteristics.

   hardware dict keys:
     - peak_flops: peak FLOPS of the hardware (e.g. 312e12 for A100)
     - peak_bandwidth_bytes_per_sec: peak memory bandwidth in bytes/sec
     - sm_count: number of streaming multiprocessors
     - cache_line_bytes: cache line size in bytes
     - tile_m: tile height for wave quantization analysis
     - tile_n: tile width for wave quantization analysis

   Return format:
   {
       "flops": int,                    # 2 * M * K * N
       "memory_bytes": int,             # (M*K + K*N + M*N) * dtype_bytes
       "arithmetic_intensity": float,   # flops / memory_bytes
       "is_compute_bound": bool,        # AI > peak_flops / peak_bandwidth_bytes_per_sec
       "k_aligned": bool,              # K % (cache_line_bytes // dtype_bytes) == 0
       "n_aligned": bool,              # N % (cache_line_bytes // dtype_bytes) == 0
       "k_pad_to": int,               # smallest multiple of cache_line_elems >= K
       "n_pad_to": int,               # smallest multiple of cache_line_elems >= N
       "tile_grid": int,              # ceil(M / tile_m) * ceil(N / tile_n)
       "wave_count": int,             # ceil(tile_grid / sm_count)
       "wave_efficiency": float,      # tile_grid / (wave_count * sm_count)
   }
"""

import torch
import torch.nn as nn
from typing import Dict, Any
