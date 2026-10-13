"""decision_loop: candidates K-way steps, calibration split, and the calibrated final checkpoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from fireworks.training.sdk.client import qualify_snapshot_name
from fireworks.training.sdk.training_spec import CosineSchedule, compute_lr, normalize_lr_scheduler_spec
from training.recipes import decision_loop, sft_loop
from training.renderer.muse_glimmer import MuseGlimmerOptions, MuseGlimmerRenderer
from training.utils.checkpoints import TrainingCheckpoints, _RUN_ID_RE, _public_logical_name
from training.utils.runner import DatasetError, UserConfigError

from .test_decision_rows import ROW, _GlimmerTokenizer
from .test_sft_loop import _make_stub_render_worker_state


def _rows(n: int) -> list[dict]:
    """ROW with alternating labels, so each question type sees more than one answer."""
    labels = [{"category": "build", "risk": 1, "unsafe": False}, {"category": "read", "risk": 0, "unsafe": True}]
    return [
        {
            "state": f"s{i}",
            "questions": {q: {**spec, "label": labels[i % 2][q]} for q, spec in ROW["questions"].items()},
        }
        for i in range(n)
    ]


def _write(path, rows) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _assert_trainer_accepts(datum) -> None:
    """The trainer's candidates contract: only the last row has candidates, K >= 2 distinct ids, gold among them."""
    rows, k = datum.model_input.length, datum.loss_fn_inputs["candidate_tokens"].shape[1]
    candidates = datum.loss_fn_inputs["candidate_tokens"].to_numpy().reshape(rows, k).tolist()
    assert all(t == -100 for row in candidates[:-1] for t in row)
    assert len(set(candidates[-1])) == k >= 2
    assert datum.loss_fn_inputs["target_tokens"].to_numpy().tolist()[-1] in candidates[-1]
    weights = datum.loss_fn_inputs["weights"].to_numpy().tolist()
    assert weights[-1] > 0 and not any(weights[:-1])


class _ControlPlane:
    """Fake service, trainer client, and control-plane checkpoint list, keyed by the names the loop saves."""

    accelerator_type = accelerator_count = training_profile = None
    trainer_job_id = "trainer-1"
    max_context_length = 4096

    def __init__(self, *, run_id: str | None = None, session_suffix: str | None = None) -> None:
        self.rows: list[dict] = []
        self.sampler_metadata: dict[str, object] = {}
        self.promoted: list[str] = []
        self.train_steps = 0
        self.step_tokens: list[int] = []
        self.tokenizer_args: tuple | None = None
        self.learning_rates: list[float] = []
        self.provisioned = False
        self.run_id = run_id
        self.session_suffix = session_suffix

    def _stored_name(self, logical: str, *, sampler: bool) -> str:
        """Publish the id the control plane lists: ``run-<32 hex>-<name>[-<8 hex>]``."""
        if self.run_id is None:
            return logical
        stored = logical
        if sampler:
            if self.session_suffix is None:
                raise AssertionError("session_suffix is required to publish a sampler checkpoint id")
            stored = qualify_snapshot_name(self.session_suffix, logical)
        return f"{self.run_id}-{stored}"

    def _row(self, name: str, checkpoint_type: str, promotable: bool) -> None:
        if self.run_id is None:
            resource = f"accounts/a/trainers/trainer-1/checkpoints/{name}"
        else:
            resource = f"accounts/a/trainingSessions/ts-{'a' * 32}/checkpoints/{name}"
        self.rows.append(
            {
                "name": resource,
                "checkpointType": checkpoint_type,
                "promotable": promotable,
                "createTime": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )

    def create_training_client(self, *args, **kwargs):
        return object()

    def close(self):
        pass

    def submit_forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
        assert loss_fn == "cross_entropy"
        assert loss_fn_config == {"target_logprob_support": "candidates"}
        for datum in batch:
            _assert_trainer_accepts(datum)
        self.train_steps += 1
        return SimpleNamespace(result=lambda timeout=None: SimpleNamespace(metrics={"loss:sum": 1.0}))

    def submit_optim_step(self, params, **kwargs):
        self.learning_rates.append(params.learning_rate)
        return SimpleNamespace(result=lambda timeout=None: SimpleNamespace(metrics={}))

    def forward(self, batch, loss_fn):
        # Readout datums read [N, K] option targets with no candidates opt-in, one K per request.
        assert len({d.loss_fn_inputs["target_tokens"].shape[1] for d in batch}) == 1
        assert all(set(d.loss_fn_inputs) == {"target_tokens", "weights"} for d in batch)
        outputs = []
        for d in batch:
            k = d.loss_fn_inputs["target_tokens"].shape[1]
            assert len(set(d.loss_fn_inputs["target_tokens"].data[-k:])) == k, "every option code on the answer row"
            answer = [0.2 if i == 0 else -0.2 for i in range(k)]
            outputs.append({"logprobs": SimpleNamespace(data=[0.0] * (d.model_input.length - 1) * k + answer)})
        return SimpleNamespace(loss_fn_outputs=outputs)

    def save_state(self, name):
        self._row(self._stored_name(name, sampler=False), "CHECKPOINT_TYPE_TRAINING_LORA", promotable=False)

    def save_weights_for_sampler(self, name, checkpoint_type=None, metadata=None, **kwargs):
        self.sampler_metadata[name] = metadata
        self._row(self._stored_name(name, sampler=True), "CHECKPOINT_TYPE_INFERENCE_LORA", promotable=True)

    def list_checkpoints(self, trainer_id):
        return list(self.rows)

    def promote_checkpoint(self, *, name, output_model_id, base_model, hot_load_deployment_id=None):
        checkpoint_id = name.rsplit("/", 1)[-1]
        if len(checkpoint_id) > 63:
            raise RuntimeError(f"id length ({len(checkpoint_id)}) must not exceed 63 characters")
        self.promoted.append(checkpoint_id)
        return {}


def _run(
    monkeypatch,
    tmp_path,
    plane: _ControlPlane,
    *,
    resume_info=None,
    serverless_run_id: str | None = None,
    **overrides,
) -> None:
    """Run ``decision_loop.main`` on 8 rows (4 steps of 2 rows) with a 4-row calibration split."""

    class _Checkpoints(TrainingCheckpoints):
        def __init__(self, *args, **kwargs):
            if serverless_run_id is not None:
                kwargs["serverless"] = True
                kwargs["current_run_id"] = serverless_run_id
            super().__init__(*args, save_appear_timeout_s=1.0, save_stabilize_s=0.0, save_poll_s=0.01, **kwargs)

        def resume(self, **kwargs):
            return resume_info

    _decision_render_state(monkeypatch, plane)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(decision_loop, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision_loop, "validate_config", lambda *args, **kwargs: None)

    def _write_running_step(runner, *, tokens, **kwargs):
        plane.step_tokens.append(tokens)

    monkeypatch.setattr(decision_loop, "write_running_step", _write_running_step)
    def _provision(**kwargs):
        plane.provisioned = True
        return plane

    monkeypatch.setattr(decision_loop, "build_service_client", _provision)
    monkeypatch.setattr(
        decision_loop, "ReconnectableClient", SimpleNamespace(from_training_client=lambda *args, **kwargs: plane)
    )
    monkeypatch.setattr(decision_loop, "TrainingCheckpoints", _Checkpoints)
    config = dict(
        log_path=str(tmp_path / "logs"),
        dataset=_write(tmp_path / "train.jsonl", _rows(8)),
        evaluation_dataset=_write(tmp_path / "dev.jsonl", _rows(4)),
        tokenizer_model="meta-models/Muse-Glimmer-30B",
        epochs=1,
        batch_size=2,
        lora_rank=8,
        render_workers=1,
        output_model_id="accounts/a/models/decision",
    )
    decision_loop.main(decision_loop.Config(**{**config, **overrides}))


def _decision_render_state(monkeypatch, plane: _ControlPlane | None = None) -> None:
    """sft_loop's render-worker state with the char-level Glimmer renderer (no HF tokenizer load)."""
    renderer = MuseGlimmerRenderer(_GlimmerTokenizer(), options=MuseGlimmerOptions(include_current_date=False))
    stub = _make_stub_render_worker_state(renderer=renderer, tokenizer=renderer.tokenizer)

    def populate(state, *, tokenizer_model, tokenizer_revision=None, tokenizer_trust_remote_code=None, **kwargs):
        if plane is not None:
            plane.tokenizer_args = (tokenizer_model, tokenizer_revision, tokenizer_trust_remote_code)
        stub(state, tokenizer_model=tokenizer_model, **kwargs)

    monkeypatch.setattr(sft_loop, "_worker_state", {})
    monkeypatch.setattr(sft_loop, "populate_render_worker_state", populate)
    monkeypatch.setattr(sft_loop, "_resolved_renderer_name", lambda **kwargs: "muse_glimmer")


def _assert_calibrated(plane: _ControlPlane, name: str) -> None:
    calibrated = plane.sampler_metadata[name]
    assert calibrated is not None
    temperature = calibrated.decision_calibration.temperature
    assert set(temperature) == {"choice", "noul", "score"} and all(0.25 <= t <= 4.0 for t in temperature.values())


def test_trains_candidates_and_promotes_the_calibrated_checkpoint(monkeypatch, tmp_path) -> None:
    plane = _ControlPlane()
    plane.save_weights_for_sampler("step-4")  # a promotable step-4 without calibration, e.g. a weight sync
    _run(monkeypatch, tmp_path, plane)

    assert plane.train_steps == 4
    assert len(plane.step_tokens) == 4 and all(tokens > 0 for tokens in plane.step_tokens)
    assert plane.sampler_metadata["step-4"] is None
    _assert_calibrated(plane, "step-4-cal")
    assert plane.promoted == ["step-4-cal"]
    assert [r["name"].rsplit("/", 1)[-1] for r in plane.rows if not r["promotable"]] == ["step-4"]


@pytest.mark.parametrize("existing", [None, "already-calibrated"], ids=["missing", "present"])
def test_resume_at_the_final_step_writes_a_missing_calibrated_checkpoint_and_promotes(
    monkeypatch, tmp_path, existing
) -> None:
    plane = _ControlPlane()
    if existing:
        plane.save_weights_for_sampler("step-4-cal", metadata=existing)
    _run(monkeypatch, tmp_path, plane, resume_info=SimpleNamespace(step=4, data_consumed=8))

    assert plane.train_steps == 0 and list(plane.sampler_metadata) == ["step-4-cal"]
    assert not [r for r in plane.rows if not r["promotable"]], "the resumable step-4 already exists"
    if existing:
        assert plane.sampler_metadata["step-4-cal"] == existing
    else:
        _assert_calibrated(plane, "step-4-cal")
    # A crash between the calibrated save and promotion still publishes on resume; re-promotion is idempotent.
    assert plane.promoted == ["step-4-cal"]


def test_pinned_tokenizer_settings_reach_the_tokenizer(monkeypatch, tmp_path) -> None:
    plane = _ControlPlane()
    _run(monkeypatch, tmp_path, plane, tokenizer_revision="rev-a", tokenizer_trust_remote_code=False)
    assert plane.tokenizer_args == ("meta-models/Muse-Glimmer-30B", "rev-a", False)


_CHAT = [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}] * 4
_UNLABELED = [{"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}] * 3


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    [
        ("save_final_checkpoint", False, UserConfigError, "requires save_final_checkpoint=True"),
        ("epochs", 0, UserConfigError, "requires epochs >= 1"),
        ("train_on_what", "last_assistant_message", UserConfigError, "train_on_what does not apply"),
        # Without evaluation_dataset, 8 rows at the default 10% carve-out ratio hold out nothing.
        ("evaluation_dataset", "", DatasetError, r"need a calibration split.*carve-out.*is empty"),
        ("evaluation_dataset", [], DatasetError, r"^evaluation_dataset .* has no rows"),
        ("evaluation_dataset", _UNLABELED, DatasetError, "calibration split has no labeled decision questions"),
        ("dataset", _CHAT, DatasetError, "^dataset row 0 is not a decision row"),
        ("evaluation_dataset", _CHAT, DatasetError, "^evaluation_dataset row 0 is not a decision row"),
    ],
)
def test_invalid_runs_are_rejected_before_training(monkeypatch, tmp_path, field, value, error, match) -> None:
    plane = _ControlPlane()
    if isinstance(value, list):
        value = _write(tmp_path / "override.jsonl", value)
    with pytest.raises(error, match=match):
        _run(monkeypatch, tmp_path, plane, **{field: value})
    assert not plane.provisioned and plane.train_steps == 0 and plane.rows == []


def test_unlabeled_carve_out_is_rejected_before_provisioning(monkeypatch, tmp_path) -> None:
    """Labeled training rows do not stand in for a calibration split whose held-out rows are all unlabeled."""
    rows = _rows(40)
    for i in decision_loop._carve_out(decision_loop.Config(log_path=""), 40):
        questions = rows[i]["questions"]
        rows[i] = {**rows[i], "questions": {q: {k: v for k, v in s.items() if k != "label"} for q, s in questions.items()}}
    plane = _ControlPlane()
    with pytest.raises(DatasetError, match="calibration split has no labeled decision questions"):
        _run(monkeypatch, tmp_path, plane, dataset=_write(tmp_path / "t.jsonl", rows), evaluation_dataset="")
    assert not plane.provisioned


def test_carve_out_holds_out_a_seeded_calibration_split(monkeypatch, tmp_path) -> None:
    _decision_render_state(monkeypatch)
    sft_loop._init_render_worker("t", "muse_glimmer", "all_assistant_messages", 4096)
    cfg = decision_loop.Config(log_path="", dataset=_write(tmp_path / "t.jsonl", _rows(40)), eval_carve_ratio=0.25)
    training_ds, eval_types, eval_data = decision_loop.prepare_datasets(cfg)
    assert len(training_ds) == 30 and len(eval_data) == len(eval_types) == 10 * 3
    _, again_types, again_data = decision_loop.prepare_datasets(cfg)
    assert again_types == eval_types
    assert [d.model_input.to_ints() for d in again_data] == [d.model_input.to_ints() for d in eval_data]


def test_decision_loop_takes_sft_loop_config() -> None:
    assert decision_loop.Config is sft_loop.Config


def test_rows_render_lazily_through_the_sft_loop_dataloader(monkeypatch, tmp_path) -> None:
    """A large row set is rendered a batch at a time by sft_loop's DataLoader, never all at once."""
    plane = _ControlPlane()
    rendered: list[int] = []
    at_first_step: list[int] = []
    loaders: list = []
    render_train_row = decision_loop.render_train_row
    make_render_dataloader = decision_loop.make_render_dataloader

    def counting_render(row):
        rendered.append(1)
        return render_train_row(row)

    def spy_loader(dataset, **kwargs):
        loaders.append(kwargs)
        return make_render_dataloader(dataset, **kwargs)

    submit = plane.submit_forward_backward

    def first_step_probe(batch, *args, **kwargs):
        if not at_first_step:
            at_first_step.append(len(rendered))
        return submit(batch, *args, **kwargs)

    monkeypatch.setattr(decision_loop, "render_train_row", counting_render)
    monkeypatch.setattr(decision_loop, "make_render_dataloader", spy_loader)
    monkeypatch.setattr(plane, "submit_forward_backward", first_step_probe)
    _run(monkeypatch, tmp_path, plane, dataset=_write(tmp_path / "big.jsonl", _rows(200)), pipeline_depth=1)

    assert len(loaders) == 1 and loaders[0]["batch_size"] == 2 and loaders[0]["shuffle"] is True
    assert at_first_step == [2], "only the first batch is rendered before the first step"
    assert len(rendered) == 200 and plane.train_steps == 100


@pytest.mark.parametrize(
    ("overrides", "spec", "legacy_warmup"),
    [
        (
            {"lr_scheduler": CosineSchedule(warmup_steps=1, min_lr_ratio=0.1)},
            CosineSchedule(warmup_steps=1, min_lr_ratio=0.1),
            0,
        ),
        ({"warmup_steps": 2}, None, 2),
    ],
    ids=["cosine", "legacy_warmup_steps"],
)
def test_lr_schedule_applies_as_in_sft_loop(monkeypatch, tmp_path, overrides, spec, legacy_warmup) -> None:
    plane = _ControlPlane()
    _run(monkeypatch, tmp_path, plane, learning_rate=1e-3, **overrides)
    schedule = normalize_lr_scheduler_spec(spec, legacy_warmup_steps=legacy_warmup)
    expected = [compute_lr(schedule, step=s, base_lr=1e-3, total_steps=4) for s in range(1, 5)]
    assert plane.learning_rates == pytest.approx(expected)
    assert len(set(plane.learning_rates)) > 1


def test_checkpoint_intervals_apply_as_in_sft_loop(monkeypatch, tmp_path) -> None:
    plane = _ControlPlane()
    _run(monkeypatch, tmp_path, plane, sampler_save_interval=2, dcp_save_interval=3)

    names = lambda promotable: [
        r["name"].rsplit("/", 1)[-1] for r in plane.rows if r["promotable"] is promotable
    ]  # noqa: E731
    assert names(True) == ["step-2", "step-4", "step-4-cal"]
    assert names(False) == ["step-3", "step-4"]
    assert plane.sampler_metadata["step-2"] is None and plane.sampler_metadata["step-4"] is None
    _assert_calibrated(plane, "step-4-cal")
    assert plane.promoted == ["step-4-cal"]


def test_render_samples_are_written_as_in_sft_loop(monkeypatch, tmp_path) -> None:
    plane = _ControlPlane()
    samples = tmp_path / "render_samples.jsonl"
    _run(monkeypatch, tmp_path, plane, render_samples_file=str(samples), render_samples_limit=3)
    records = [json.loads(line) for line in samples.read_text().splitlines()]
    assert len(records) == 3
    assert all(r["training_loss_weights"][-1] == 1.0 and not any(r["training_loss_weights"][:-1]) for r in records)


def test_a_save_carrying_metadata_is_never_skipped(tmp_path) -> None:
    plane = _ControlPlane()
    ckpt = TrainingCheckpoints(plane, plane, trainer_id="trainer-1", log_path=str(tmp_path), save_poll_s=0.01)
    for _ in range(2):  # an ordinary save of an existing sampler checkpoint is skipped, as before
        ckpt.save("step-4", resumable=False, promotable=True)
    assert len(plane.rows) == 1 and ckpt.has_promotable("step-4") and not ckpt.has_promotable("step-4-cal")
    with pytest.raises(RuntimeError, match="sampler_metadata is never skipped"):
        ckpt.save("step-4", resumable=False, promotable=True, sampler_metadata=SimpleNamespace())


# Serverless publication of the canary's checkpoint: run- + 32 hex, then the SDK's 8-hex suffix.
_INCIDENT_RUN_ID = "run-0bc3603413bb41a4aa2fb101efef4e56"
_INCIDENT_SESSION_SUFFIX = "16439373"


def _published_sampler_id(
    logical: str,
    *,
    run_id: str = _INCIDENT_RUN_ID,
    session_suffix: str = _INCIDENT_SESSION_SUFFIX,
) -> str:
    assert _RUN_ID_RE.fullmatch(run_id)
    return f"{run_id}-{qualify_snapshot_name(session_suffix, logical)}"


@pytest.mark.parametrize("step", [10, 25, 100, 10_000, 99_999_999])
def test_calibrated_name_stays_inside_the_promoted_id_budget(step: int) -> None:
    name = decision_loop.calibrated_checkpoint_name(step)
    assert name == f"step-{step}-cal"
    assert len(name) <= decision_loop.PROMOTED_CHECKPOINT_NAME_BUDGET == 17
    public = _published_sampler_id(name)
    assert len(public) <= 63
    assert _public_logical_name(public) == name


def test_eight_digit_steps_fill_the_budget_and_the_next_digit_does_not() -> None:
    assert len(decision_loop.calibrated_checkpoint_name(99_999_999)) == 17
    assert len(_published_sampler_id("step-99999999-cal")) == 63
    with pytest.raises(UserConfigError, match="allows 17"):
        decision_loop.calibrated_checkpoint_name(100_000_000)


def test_legacy_calibrated_suffix_is_one_over_the_promoted_id_cap() -> None:
    legacy = _published_sampler_id("step-25-calibrated")
    assert legacy == "run-0bc3603413bb41a4aa2fb101efef4e56-step-25-calibrated-16439373"
    assert len(legacy) == 64
    assert len(_published_sampler_id(decision_loop.calibrated_checkpoint_name(25))) == 57


def test_ten_step_calibrated_checkpoint_promotes_under_the_run_prefix(monkeypatch, tmp_path) -> None:
    """A 10-step decision run promotes the calibrated sampler id the gateway will accept."""
    plane = _ControlPlane(run_id=_INCIDENT_RUN_ID, session_suffix=_INCIDENT_SESSION_SUFFIX)
    _run(
        monkeypatch,
        tmp_path,
        plane,
        dataset=_write(tmp_path / "train10.jsonl", _rows(20)),
        batch_size=2,
        serverless_run_id=_INCIDENT_RUN_ID,
    )
    public = _published_sampler_id("step-10-cal")
    assert plane.train_steps == 10
    assert plane.promoted == [public]
    assert len(public) <= 63
    assert _public_logical_name(public) == "step-10-cal"
    _assert_calibrated(plane, "step-10-cal")


def test_step_25_calibrated_checkpoint_promotes_under_the_incident_prefix(monkeypatch, tmp_path) -> None:
    """Resume at step 25, the canary's step, still promotes a 63-or-shorter id."""
    plane = _ControlPlane(run_id=_INCIDENT_RUN_ID, session_suffix=_INCIDENT_SESSION_SUFFIX)
    _run(
        monkeypatch,
        tmp_path,
        plane,
        dataset=_write(tmp_path / "train25.jsonl", _rows(50)),
        batch_size=2,
        resume_info=SimpleNamespace(step=25, data_consumed=50),
        serverless_run_id=_INCIDENT_RUN_ID,
    )
    public = _published_sampler_id("step-25-cal")
    assert plane.train_steps == 0
    assert plane.promoted == [public]
    assert len(public) == 57
    assert _public_logical_name(public) == "step-25-cal"
