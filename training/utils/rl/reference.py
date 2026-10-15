"""Optional frozen-reference scoring for custom RL train steps.

LoRA can forward the frozen base on its policy trainer with adapters disabled.
Full-parameter tuning needs an independent frozen base deployment: never use
the policy rollout deployment, whose weights change after publication.
Provisioning and resource ownership stay with the SDK and the caller.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Protocol

from fireworks.training import sdk

from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.rl.losses import PolicyBatch
from training.utils.timer import elapsed_timer


class ReferenceForward(Protocol):
    def forward(self, datums: list[sdk.Datum], loss_fn: str) -> Any: ...


class LoraReference:
    """Blocking forward on an SDK-created, base-only training client.

    Create that client once with
    ``service.create_reference_client(policy_client=training_client)`` on a LoRA
    service without explicit reference shape/job overrides. The SDK shares the
    policy trainer and disables its adapter. Do not wrap the policy client itself
    or use this shared-base path for full-parameter training.
    """

    def __init__(self, base_client: Any, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._base_client = base_client
        self._timeout = timeout

    def forward(self, datums: list[sdk.Datum], loss_fn: str = "cross_entropy") -> Any:
        # The frozen base chooses its own experts and full-vocabulary scores.
        # Replaying the policy's routes or sampled support changes that baseline.
        reference_data = [
            replace(datum, model_input=sdk.ModelInput(chunks=datum.model_input.chunks)) for datum in datums
        ]
        return self._base_client.forward(reference_data, loss_fn).result(timeout=self._timeout)


class SamplerReference:
    """Score a frozen inference sampler in the trainer-forward token layout.

    For full-parameter tuning, pass a sampling client for a separate base-model
    deployment, with the same tokenizer and sufficient context. Never hot-load
    policy weights into it. A frozen serverless snapshot is also supported.

    ``compute_logprobs`` scores the supplied tokens at temperature 1 with no
    sampling truncation. Calls submit/wait per sequence so the caller can bound
    concurrency across batches. This adapter supports shifted text datums only.
    """

    def __init__(self, sampler: Any, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._sampler = sampler
        self._timeout = timeout

    def forward(self, datums: list[sdk.Datum], loss_fn: str = "cross_entropy") -> SimpleNamespace:
        if loss_fn != "cross_entropy":
            raise ValueError("sampler reference supports only cross_entropy scoring")
        outputs = []
        for datum in datums:
            targets = list(datum.loss_fn_inputs["target_tokens"].data)
            tokens = datum.model_input.to_ints()
            if not targets:
                raise ValueError("reference datum must contain target tokens")
            if len(tokens) != len(targets) or tokens[1:] != targets[:-1]:
                raise ValueError("sampler reference requires shifted text target tokens")
            # Restore the final target omitted from the trainer's input. Prompt
            # logprob position 0 has no preceding token; position i+1 scores
            # trainer target i. Decode/re-encode would change this alignment.
            sequence = sdk.ModelInput.from_ints([*tokens, int(targets[-1])])
            values = self._sampler.compute_logprobs(sequence).result(timeout=self._timeout)
            if len(values) != len(targets) + 1:
                raise ValueError("reference logprob length does not match target tokens")
            aligned = values[1:]
            if any(value is None or not math.isfinite(value) for value in aligned):
                raise ValueError("reference sampler returned an unscored target token or non-finite logprob")
            outputs.append({"logprobs": SimpleNamespace(data=aligned)})
        return SimpleNamespace(loss_fn_outputs=outputs)


def with_reference_logprobs(batch: PolicyBatch, reference: ReferenceForward) -> PolicyBatch:
    """Return a batch with frozen-reference scores; preserve its clipping anchor.

    Both adapters block. In an async custom loop, call this with
    ``await asyncio.to_thread(with_reference_logprobs, batch, reference)``
    before ``forward_backward_custom``. Resource cleanup remains caller-owned.
    """
    with elapsed_timer("reference_forward"):
        result = reference.forward(batch.data, "cross_entropy")
    rows = [list(output["logprobs"].data) for output in result.loss_fn_outputs]
    if len(rows) != len(batch.data) or any(
        len(row) != len(datum.loss_fn_inputs["target_tokens"].data)
        or any(value is None or not math.isfinite(value) for value in row)
        for row, datum in zip(rows, batch.data, strict=True)
    ):
        raise ValueError("Reference logprobs must be finite and align with target tokens")
    return replace(batch, ref_logprobs=rows)
