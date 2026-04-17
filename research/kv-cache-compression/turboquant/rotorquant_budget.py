"""
Bit-budget accounting helpers for RotorQuant two-term (MSE + QJL) scoring.

The fused two-term path stores **one** MSE Lloyd index per logical head
dimension (trimmed layout): slots ``0..d-1`` map to group/blade, omitting the
unused Clifford padding when ``ceil(d/3)*3 > d``. Physical effective bits/dim
therefore counts ``d`` MSE slots, matching per-dimension counting in TurboQuant
rows.

Legacy helpers :func:`vector_grade_slot_count` and
:func:`mse_slot_excess_over_head_dim` still describe the **untrimmed**
``ceil(d/3)*3`` slot count (useful when comparing to older layouts or
``RotorQuantProd``). :func:`clifford_padded_slots_beyond_head_dim` is the
difference between that padded count and ``d``.

TurboQuant rows use ``bits + 16/d`` (MSE indices per ``d``, QJL signs, fp16
residual norm). :func:`turboquant_row_effective_bits_per_dim` matches that
formula for apples-to-apples comparison with trimmed RotorQuant storage.
"""

from __future__ import annotations

from typing import TypedDict


class PhysicalGapDecomposition(TypedDict):
    """Closed-form split of ``physical_effective - integer_target_bits`` (two-term RotorQuant)."""

    physical_effective_bits_per_dim: float
    integer_target_bits: float
    gap_physical_vs_integer_target: float
    component_residual_norm_surplus: float
    component_mse_slot_padding_surplus: float
    gap_decomposition_sum: float
    gap_decomposition_abs_error: float


def mv_group_count(head_dim: int) -> int:
    """Number of Cl(3,0) embedding groups; matches ``RotorQuantMSE.n_groups``."""
    return (head_dim + 2) // 3


def vector_grade_slot_count(head_dim: int) -> int:
    """MSE index slots for the vector-only layout (e1,e2,e3 per group)."""
    return mv_group_count(head_dim) * 3


def mse_slot_excess_over_head_dim(head_dim: int) -> int:
    """Non-negative; zero iff ``head_dim`` is a multiple of 3 (untrimmed grouped layout)."""
    return vector_grade_slot_count(head_dim) - head_dim


def stored_mse_index_slots(head_dim: int) -> int:
    """MSE index slots stored by the trimmed fused path (one per logical dimension)."""
    return head_dim


def clifford_padded_slots_beyond_head_dim(head_dim: int) -> int:
    """Slots in the untrimmed ``n_groups*3`` layout that are not stored after trimming."""
    return mse_slot_excess_over_head_dim(head_dim)


def physical_two_term_effective_bits_per_dim(head_dim: int, bits: int) -> float:
    """
    Storage-accurate effective bits/dim for the trimmed Triton two-term path:

    ``(d * mse_bits + d + 16) / d = mse_bits + 1 + 16/d``

    where ``mse_bits = max(bits - 1, 1)`` (stage-1 Lloyd-Max), the middle ``d``
    is QJL sign storage (1 bit/slot in the harness convention), and ``+16`` is
    the fp16 residual norm channel — same structure as the TurboQuant row
    ``bits + 16/d`` when the integer row label matches ``mse_bits + 1``.
    """
    mse_bits = max(bits - 1, 1)
    d = stored_mse_index_slots(head_dim)
    return (d * mse_bits + d + 16.0) / d


def turboquant_row_effective_bits_per_dim(head_dim: int, bits: int) -> float:
    """Matches ``run_kernel_table`` TurboQuant row: ``bits + 16/head_dim``."""
    return bits + 16.0 / head_dim


def physical_minus_turboquant_row_bits(
    head_dim: int, bits: int, *, physical: float | None = None
) -> float:
    """Surplus of physical accounting over TurboQuant-row comparison."""
    phys = (
        physical
        if physical is not None
        else physical_two_term_effective_bits_per_dim(head_dim, bits)
    )
    return phys - turboquant_row_effective_bits_per_dim(head_dim, bits)


def fp16_residual_norm_bits_per_dim(head_dim: int) -> float:
    """Bits per head dimension from one scalar fp16 residual-norm channel (harness convention)."""
    return 16.0 / head_dim


def decompose_physical_gap_vs_integer_target(
    head_dim: int, bits: int, *, physical: float | None = None
) -> PhysicalGapDecomposition:
    """
    Explain ``physical_effective_bits_per_dim - bits`` (integer row label) without conflating sources.

    **component_residual_norm_surplus** is ``(bits + 16/d) - bits = 16/d``: the two-term rows always
    carry a scalar fp16 norm amortized over ``d`` logical dimensions, so comparing physical
    effective bits to the integer ``bits`` label picks up this channel even when representation
    is otherwise fair.

    **component_mse_slot_padding_surplus** is ``physical - (bits + 16/d)``, i.e.
    :func:`physical_minus_turboquant_row_bits`. It is **zero** when physical
    storage uses the trimmed ``d`` MSE slots (current fused path); non-zero
    only if a caller passes a custom ``physical`` that still counts padded slots.

    For harness rows, ``component_residual_norm_surplus + component_mse_slot_padding_surplus``
    equals ``physical - bits`` (``gap_decomposition_abs_error`` should be ~0).
    """
    phys = (
        physical
        if physical is not None
        else physical_two_term_effective_bits_per_dim(head_dim, bits)
    )
    tq = turboquant_row_effective_bits_per_dim(head_dim, bits)
    residual = fp16_residual_norm_bits_per_dim(head_dim)
    padding = phys - tq
    raw_delta = phys - float(bits)
    explained = residual + padding
    return PhysicalGapDecomposition(
        physical_effective_bits_per_dim=phys,
        integer_target_bits=float(bits),
        gap_physical_vs_integer_target=max(raw_delta, 0.0),
        component_residual_norm_surplus=residual,
        component_mse_slot_padding_surplus=padding,
        gap_decomposition_sum=explained,
        gap_decomposition_abs_error=abs(explained - raw_delta),
    )
