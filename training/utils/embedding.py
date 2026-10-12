"""Bidirectional in-batch InfoNCE for embedding (retrieval) fine-tuning.

A batch holds ``B`` (query, positive) pairs laid out as ``2B`` sequences,
``[Q_0..Q_{B-1}, D_0..D_{B-1}]``. Each query's paired document is its only
positive; the other ``B-1`` documents are in-batch negatives, and vice versa.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def bidirectional_info_nce_loss(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional in-batch InfoNCE over ``B`` (query, positive) pairs.

    Row ``i`` of each tensor is paired (query ``i`` <-> positive ``i``). Every
    other document in the batch is an in-batch negative.
    """
    z_q = F.normalize(query_embeddings, p=2, dim=-1)
    z_d = F.normalize(doc_embeddings, p=2, dim=-1)
    sim = (z_q @ z_d.t()) / temperature  # [B, B]
    targets = torch.arange(sim.shape[0], device=sim.device)
    loss_q2d = F.cross_entropy(sim, targets)
    loss_d2q = F.cross_entropy(sim.t(), targets)
    loss = 0.5 * (loss_q2d + loss_d2q)
    with torch.no_grad():
        recall_at_1 = (sim.argmax(dim=1) == targets).float().mean().item()
        metrics = {
            "loss": float(loss.item()),
            "loss_q2d": float(loss_q2d.item()),
            "loss_d2q": float(loss_d2q.item()),
            "in_batch_recall_at_1": recall_at_1,
        }
    return loss, metrics


def info_nce_loss_fn(
    num_queries: int,
    temperature: float,
) -> Callable[[Any, Sequence[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]]:
    """Client loss for ``forward_backward_custom(..., output="embedding")``.

    The trainer returns one pooled vector per sequence; the first
    ``num_queries`` are queries and the next ``num_queries`` their positives.
    """

    def loss_fn(_data: Any, embeddings: Sequence[torch.Tensor]):
        queries = torch.stack(list(embeddings[:num_queries]), dim=0).to(torch.float32)
        docs = torch.stack(list(embeddings[num_queries : 2 * num_queries]), dim=0).to(torch.float32)
        return bidirectional_info_nce_loss(queries, docs, temperature=temperature)

    return loss_fn
