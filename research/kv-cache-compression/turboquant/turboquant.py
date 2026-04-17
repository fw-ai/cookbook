"""
TurboQuant: Two-stage vector quantization with near-optimal distortion.

Stage 1 (MSE): Random rotation + per-coordinate Lloyd-Max quantization
Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased inner products

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .lloyd_max import LloydMaxCodebook


def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate a random orthogonal rotation matrix via QR decomposition of Gaussian matrix.
    This is the Haar-distributed random rotation used in TurboQuant.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    # Generate random Gaussian matrix and QR decompose
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    # Ensure proper rotation (det = +1) by fixing sign ambiguity in QR
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate the random projection matrix S for QJL.
    S has i.i.d. N(0,1) entries, shape (m, d).
    Default m = d (same dimensionality).
    """
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)
    return S.to(device)


class TurboQuantMSE(nn.Module):
    """
    Stage 1: MSE-optimal quantizer.
    Randomly rotates, then applies per-coordinate Lloyd-Max quantization.
    """

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        # Precompute rotation matrix (Pi_T stored explicitly so torch.compile avoids .T on nn.Parameter/buffer weakrefs).
        Pi = generate_rotation_matrix(d, seed=seed, device=device)
        self.register_buffer("Pi", Pi)
        self.register_buffer("Pi_T", Pi.mT.contiguous())

        # Precompute Lloyd-Max codebook
        self.codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", self.codebook.centroids.to(device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation: y = Pi @ x (as x @ Pi^T)."""
        # x: (batch, d) or (d,)
        return x @ self.Pi_T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Undo rotation: x = Pi^T @ y."""
        return y @ self.Pi

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors to codebook indices. Returns integer indices."""
        y = self.rotate(x)
        c = self.centroids
        d_dim = y.shape[-1]
        rest = y.shape[:-1]
        flat = y.reshape(-1)
        L = c.numel()
        # Sorted 1-D Lloyd–Max centroids: NN is one of the two insertion neighbors.
        # Matches full argmin (including tie-break as min index) without O(n_levels) slabs.
        i = torch.searchsorted(c, flat, right=False)
        i0 = (i - 1).clamp(0, L - 1)
        i1 = i.clamp(0, L - 1)
        d0 = (flat - c[i0]).abs()
        d1 = (flat - c[i1]).abs()
        pick = torch.where(d0 < d1, i0, torch.where(d1 < d0, i1, torch.minimum(i0, i1)))
        return pick.reshape(*rest, d_dim)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to vectors."""
        y_hat = self.centroids[indices]  # (..., d)
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full quantize-dequantize cycle.
        Returns: (reconstructed_x, indices)
        """
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class TurboQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.
    Uses (b-1)-bit MSE quantizer + 1-bit QJL on residuals.

    Total storage per vector: (b-1)*d bits for MSE indices + d bits for QJL signs + 16 bits for residual norm
    Effective: ~b bits per dimension (the QJL bit replaces one MSE bit)
    """

    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None, seed: int = 42, device: str = "cpu"):
        """
        Args:
            d: vector dimension
            bits: total bit budget per coordinate (MSE uses bits-1, QJL uses 1)
            qjl_dim: projection dimension for QJL (default = d)
            seed: random seed for reproducibility
            device: torch device
        """
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        # Stage 1: MSE quantizer with (bits-1) bits
        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device)

        # Stage 2: QJL projection matrix (S_T explicit for torch.compile on quantize).
        S = generate_qjl_matrix(d, m=self.qjl_dim, seed=seed + 1, device=device)
        self.register_buffer("S", S)
        self.register_buffer("S_T", S.mT.contiguous())
        # Hoist scalar used in inner_product paths (same float as local sqrt(pi/2)/m each call).
        self._qjl_correction_scale: float = math.sqrt(math.pi / 2) / float(self.qjl_dim)

    @torch.inference_mode()
    def quantize(self, x: torch.Tensor) -> dict:
        """
        Full TurboQuant quantization.

        Returns dict with:
            - 'mse_indices': (batch, d) int tensor, MSE codebook indices
            - 'qjl_signs': (batch, qjl_dim) sign bits of QJL-projected residual
            - 'residual_norm': (batch,) L2 norm of residual
        """
        # Stage 1: same numerics as ``self.mse(x)`` but skip ``nn.Module.__call__`` overhead.
        mse_indices = self.mse.quantize(x)
        x_hat = self.mse.dequantize(mse_indices)

        # Compute residual
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)  # (batch, 1)

        # Stage 2: QJL - project residual and take sign
        projected = residual @ self.S_T  # (batch, qjl_dim)
        qjl_signs = torch.sign(projected)  # (batch, qjl_dim)
        qjl_signs[qjl_signs == 0] = 1.0  # map zeros to +1

        return {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }

    @torch.inference_mode()
    def dequantize(self, compressed: dict) -> torch.Tensor:
        """Dequantize MSE component (for reconstruction)."""
        return self.mse.dequantize(compressed["mse_indices"])

    @torch.inference_mode()
    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute unbiased inner product estimate: <y, x> using compressed representation of x.

        The estimator is:
            <y, x_mse> + ||r|| * sqrt(pi/2) / m * <S @ y, qjl_signs>

        Args:
            y: query vectors (batch, d) or (d,)
            compressed: dict from quantize()

        Returns:
            Estimated inner products (batch,)
        """
        # Term 1: inner product with MSE reconstruction
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        term1 = (y * x_mse).sum(dim=-1)

        # Term 2: QJL correction
        # Project query with same S matrix (but don't quantize query)
        y_projected = y @ self.S_T  # (batch, qjl_dim)
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)

        term2 = compressed["residual_norm"] * self._qjl_correction_scale * qjl_ip

        return term1 + term2

    @torch.inference_mode()
    def inner_product_queries(
        self, y: torch.Tensor, compressed: dict, y_projected: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Batched unbiased inner products for many queries against one compressed key matrix.

        Args:
            y: (Q, d) query vectors (float tensor on the same device as compressed).
            compressed: dict from ``quantize`` on a key matrix of shape (L, d).
            y_projected: optional precomputed ``y @ S_T`` with shape (Q, qjl_dim); avoids an
                extra GEMM when the caller already projected a larger query batch.

        Returns:
            (Q, L) where entry [q, ell] matches ``inner_product(y[q], row_ell_compressed)``
            with the same estimator as :meth:`inner_product`.
        """
        if y.ndim == 1:
            y = y.unsqueeze(0)
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        term1 = torch.matmul(y, x_mse.transpose(-2, -1))

        if y_projected is None:
            y_projected = y @ self.S_T
        signs = compressed["qjl_signs"]
        qjl_ip = torch.matmul(y_projected, signs.transpose(-2, -1))

        rnorm = compressed["residual_norm"]
        term2 = rnorm.unsqueeze(0) * self._qjl_correction_scale * qjl_ip
        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict:
        """Quantize input vectors."""
        return self.quantize(x)


class TurboQuantKVCache:
    """
    KV cache wrapper that uses TurboQuant to compress keys and values.
    Drop-in replacement concept for a standard KV cache.
    """

    def __init__(self, d_key: int, d_value: int, bits: int = 3, seed: int = 42, device: str = "cpu"):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device

        # Use TurboQuantProd for keys (need inner products for attention)
        self.key_quantizer = TurboQuantProd(d_key, bits, seed=seed, device=device)
        # Use TurboQuantMSE for values (need MSE reconstruction, not inner products)
        self.value_quantizer = TurboQuantMSE(d_value, bits, seed=seed + 100, device=device)

        # Storage
        self.key_cache = []    # list of compressed key dicts
        self.value_cache = []  # list of (indices,) tuples

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Append new key-value pairs to cache.
        keys: (batch, seq_len, d_key) or (seq_len, d_key)
        values: (batch, seq_len, d_value) or (seq_len, d_value)
        """
        orig_shape = keys.shape
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)

        compressed_keys = self.key_quantizer.quantize(flat_keys)
        value_indices = self.value_quantizer.quantize(flat_values)

        self.key_cache.append({
            "mse_indices": compressed_keys["mse_indices"],
            "qjl_signs": compressed_keys["qjl_signs"],
            "residual_norm": compressed_keys["residual_norm"],
            "shape": orig_shape,
        })
        self.value_cache.append({
            "indices": value_indices,
            "shape": values.shape,
        })

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores between queries and all cached keys.
        Uses unbiased inner product estimation via TurboQuant.

        queries: (batch, d_key) or (d_key,)
        Returns: scores for each cached position
        """
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached)
            scores.append(s)
        return torch.cat(scores, dim=-1) if scores else torch.tensor([])

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values."""
        values = []
        for cached in self.value_cache:
            v = self.value_quantizer.dequantize(cached["indices"])
            values.append(v)
        return torch.cat(values, dim=0) if values else torch.tensor([])

    def memory_usage_bits(self) -> dict:
        """Estimate memory usage in bits."""
        n_keys = sum(c["mse_indices"].numel() for c in self.key_cache) if self.key_cache else 0
        n_qjl = sum(c["qjl_signs"].numel() for c in self.key_cache) if self.key_cache else 0
        n_norms = sum(c["residual_norm"].numel() for c in self.key_cache) if self.key_cache else 0
        n_values = sum(c["indices"].numel() for c in self.value_cache) if self.value_cache else 0

        key_bits = n_keys * self.key_quantizer.mse_bits + n_qjl * 1 + n_norms * 16
        value_bits = n_values * self.bits
        fp16_equivalent = (n_keys + n_values) * 16  # what fp16 would cost

        return {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "total_bits": key_bits + value_bits,
            "fp16_bits": fp16_equivalent,
            "compression_ratio": fp16_equivalent / (key_bits + value_bits) if (key_bits + value_bits) > 0 else 0,
        }

    def __len__(self):
        return sum(c["mse_indices"].shape[0] for c in self.key_cache) if self.key_cache else 0
