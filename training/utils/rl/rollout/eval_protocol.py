"""Adapter from eval-protocol/eval3 evaluators to per-sample rollout fns.

The async RL loop only depends on ``rollout_fn(row) -> RolloutSample | None``.
This module lets managed RFT accept the user-facing eval3 shape instead:
``@evaluation_test(..., rollout_processor=..., completion_params=..., steps=...)``.
The adapter extracts the decorator metadata, invokes the rollout processor for
exactly one row per call, optionally runs the pointwise evaluator body, and
converts the resulting ``EvaluationRow`` into a token-native ``RolloutSample``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from eval_protocol.models import EPParameters, EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.utils.rl.rollout.types import RolloutSample

RolloutFn = Callable[[Any], Awaitable[RolloutSample | None]]
RolloutFnFactory = Callable[[Any], RolloutFn]
EvalRowFactory = Callable[[Any], EvaluationRow]
SampleConverter = Callable[[EvaluationRow], RolloutSample | None]
CompletionParamsFactory = Callable[[Any, EPParameters], dict[str, Any]]
ProcessorFactory = Callable[[Any, EPParameters], Any]

logger = logging.getLogger(__name__)


def get_eval_protocol_params(evaluator: Any) -> EPParameters:
    """Return ``EPParameters`` attached by ``@evaluation_test``.

    Tests and internal callers may pass an ``EPParameters`` object directly,
    but the normal path is a decorated evaluator function with ``__ep_params__``.
    """
    if isinstance(evaluator, EPParameters):
        return evaluator
    params = getattr(evaluator, "__ep_params__", None)
    if isinstance(params, EPParameters):
        return params
    raise TypeError(
        "Expected an eval-protocol @evaluation_test function or EPParameters "
        f"object, got {type(evaluator).__name__}."
    )


def load_eval_protocol_input_rows(evaluator: Any) -> list[EvaluationRow]:
    """Load static input rows from a decorated eval-protocol evaluator.

    This intentionally supports the RFT-compatible subset used for async RL:
    preconstructed ``input_rows``. Dataset paths and data loaders are still
    valid eval3 features, but managed RFT should materialize them before it
    constructs row requests so resume cursors and sharding are explicit.
    """
    params = get_eval_protocol_params(evaluator)
    input_rows = params.input_rows or []
    rows: list[EvaluationRow] = []
    for dataset_rows in input_rows:
        for row in dataset_rows or []:
            rows.append(_copy_row(row))
    if params.filtered_row_ids is not None:
        allowed = set(params.filtered_row_ids)
        rows = [row for row in rows if row.input_metadata.row_id in allowed]
    if params.max_dataset_rows is not None:
        rows = rows[: params.max_dataset_rows]
    if params.preprocess_fn is not None:
        rows = params.preprocess_fn(rows)
    return rows


def make_eval_protocol_rollout_fn_factory(
    evaluator: Any,
    *,
    row_factory: EvalRowFactory | None = None,
    sample_converter: SampleConverter,
    completion_params_factory: CompletionParamsFactory | None = None,
    processor_factory: ProcessorFactory | None = None,
    apply_evaluator: bool = True,
    setup_processor: bool = True,
    swallow_exceptions: bool = True,
) -> RolloutFnFactory:
    """Build a per-sample rollout factory from an eval3 evaluator.

    ``evaluator`` is expected to be a function decorated with
    ``@evaluation_test``. The decorator remains the user-facing place to
    consolidate rollout processor and config. The returned factory is the
    internal async-RL boundary.
    """
    params = get_eval_protocol_params(evaluator)
    if params.mode != "pointwise":
        raise ValueError(
            "eval-protocol rollout adapter currently supports only "
            f"pointwise evaluators, got mode={params.mode!r}."
        )

    eval_fn = _origin_eval_fn(evaluator) if apply_evaluator else None
    build_row = row_factory or default_eval_row_factory
    build_completion_params = completion_params_factory or default_completion_params_factory

    def factory(setup: Any) -> RolloutFn:
        processor = (
            processor_factory(setup, params)
            if processor_factory is not None
            else params.rollout_processor
        )
        if processor is None:
            raise ValueError(
                "eval-protocol evaluator has no rollout_processor. "
                "Pass rollout_processor=... to @evaluation_test or provide "
                "processor_factory=...."
            )
        if setup_processor and hasattr(processor, "setup"):
            processor.setup()

        async def rollout_fn(row: Any) -> RolloutSample | None:
            try:
                eval_row = build_row(row)
                config = RolloutProcessorConfig(
                    completion_params=build_completion_params(setup, params),
                    mcp_config_path=params.mcp_config_path or "",
                    server_script_path=params.server_script_path,
                    steps=int(params.steps),
                    logger=params.logger,
                    semaphore=asyncio.Semaphore(1),
                    kwargs=dict(params.rollout_processor_kwargs or {}),
                    exception_handler_config=params.exception_handler_config,
                )
                result = await _run_single_row_rollout(processor, eval_row, config)
                if eval_fn is not None:
                    evaluated = await _maybe_await(eval_fn(result))
                    if evaluated is not None:
                        result = evaluated
                    if not isinstance(result, EvaluationRow):
                        raise TypeError(
                            "eval-protocol evaluator must return EvaluationRow "
                            f"or None, got {type(result).__name__}."
                        )
                return sample_converter(result)
            except Exception:
                if not swallow_exceptions:
                    raise
                logger.exception("eval-protocol rollout_fn failed; dropping sample")
                return None

        return rollout_fn

    return factory


def default_eval_row_factory(row: Any) -> EvaluationRow:
    """Default row adapter for callers that already traffic in EvaluationRows."""
    if isinstance(row, EvaluationRow):
        return _copy_row(row)
    if isinstance(row, dict) and isinstance(row.get("evaluation_row"), EvaluationRow):
        return _copy_row(row["evaluation_row"])
    raise TypeError(
        "Default eval-protocol row factory expects an EvaluationRow or a dict "
        "with an 'evaluation_row' EvaluationRow. Provide row_factory=... for "
        f"{type(row).__name__} inputs."
    )


def default_completion_params_factory(setup: Any, params: EPParameters) -> dict[str, Any]:
    """Merge evaluator completion params with runtime setup sampling defaults."""
    completion_params = _first_completion_params(params.completion_params)
    setup_sample_kwargs = getattr(setup, "sample_kwargs", None) if setup is not None else None
    if isinstance(setup_sample_kwargs, dict):
        completion_params.update(setup_sample_kwargs)
    setup_model = getattr(setup, "model", None) if setup is not None else None
    if setup_model:
        completion_params["model"] = setup_model
    return completion_params


async def _run_single_row_rollout(
    processor: Any,
    row: EvaluationRow,
    config: RolloutProcessorConfig,
) -> EvaluationRow:
    tasks = processor([row], config)
    if not isinstance(tasks, Sequence) or len(tasks) != 1:
        raise ValueError(
            "eval-protocol rollout adapter expects processor([row], config) "
            f"to return exactly one task/result, got {type(tasks).__name__} "
            f"with length {len(tasks) if isinstance(tasks, Sequence) else 'unknown'}."
        )
    result = await _maybe_await(tasks[0])
    if not isinstance(result, EvaluationRow):
        raise TypeError(
            "eval-protocol rollout processor must return EvaluationRow, "
            f"got {type(result).__name__}."
        )
    return result


def _first_completion_params(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return {}
        first = value[0]
        return dict(first or {})
    return {}


def _copy_row(row: EvaluationRow) -> EvaluationRow:
    if hasattr(row, "model_copy"):
        return row.model_copy(deep=True)
    return row.copy(deep=True)


def _origin_eval_fn(evaluator: Any) -> Callable[[EvaluationRow], Any]:
    return (
        getattr(evaluator, "_origin_func", None)
        or getattr(evaluator, "__wrapped__", None)
        or evaluator
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
