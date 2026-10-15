import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from training.examples.sft import muse_glimmer_starter as starter


def test_promotion_deadline_is_end_of_oct_31_pacific():
    assert starter.promotion_active(datetime(2026, 11, 1, 6, 59, tzinfo=timezone.utc))
    assert not starter.promotion_active(datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc))


def test_serverless_names_include_a_promoted_model():
    names = starter._names("acme", "router-v1")
    assert names["serverless_model"] == "accounts/acme/models/muse-starter-sl-router-v1"
    assert starter.parse_args(["launch", "--surface", "serverless", "--no-promote"]).no_promote


def test_muse_configuration_matches_skill():
    assert starter.BASE_MODEL == "accounts/fireworks/models/muse-glimmer-30b"
    assert starter.TOKENIZER_MODEL == "meta-models/Muse-Glimmer-30B"
    assert starter.RENDERER_NAME == "muse_glimmer"
    assert starter.DEFAULT_DATASET == "onboarding-prompt-router"


def test_defaults_are_a_small_first_run():
    args = starter.parse_args(["launch"])
    assert args.surface == "managed"
    assert args.epochs == 1
    assert args.lora_rank == 8


def test_serverless_plan_prints_every_resolved_training_default(monkeypatch, capsys):
    monkeypatch.setattr(starter, "_connect", lambda: (SimpleNamespace(), "acme"))
    args = starter.parse_args(["plan", "--surface", "serverless"])
    assert starter.cmd_plan(args) == 0
    params = json.loads(capsys.readouterr().out)["parameters"]
    assert params["train_on"] == "last_assistant_turn"
    assert params["loss"] == "cross_entropy"
    assert params["optimizer"] == {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }
    assert params["emit_grad_norm_metrics"] is True
    assert params["lora_alpha"] == starter.LORA_ALPHA
    assert params["lora_targets"] == starter.LORA_TARGETS


def test_managed_launch_refuses_to_reuse_a_job_with_different_settings(monkeypatch, capsys):
    args = starter.parse_args(["launch", "--run-id", "r1", "--lora-rank", "16"])
    existing = SimpleNamespace(
        name="accounts/acme/supervisedFineTuningJobs/muse-starter-r1", state="JOB_STATE_COMPLETED",
        base_model=starter.BASE_MODEL, dataset=f"accounts/acme/datasets/{args.dataset}",
        output_model="accounts/acme/models/muse-starter-r1", epochs=args.epochs, lora_rank=8,
        learning_rate=args.learning_rate, max_context_length=args.max_context_length,
    )

    def create(**kwargs):
        raise AssertionError("must not create when the run ID exists")

    jobs = SimpleNamespace(get=lambda job_id: existing, create=create)
    monkeypatch.setattr(starter, "_connect", lambda: (SimpleNamespace(supervised_fine_tuning_jobs=jobs), "acme"))
    assert starter.cmd_launch_managed(args) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["reused"] is False
    assert out["mismatched"]["lora_rank"] == {"existing": 8, "requested": 16}


def test_serverless_prepare_splits_the_users_file(tmp_path):
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": json.dumps({"route": "small model"})}]}
            for i in range(10)]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    args = starter.parse_args(["prepare", "--run-id", "r", "--file", str(data), "--n-eval", "3",
                               "--log-path", str(tmp_path / "run")])
    run_dir, train_ex, eval_ex, _, _, report = starter._prepare(args)
    assert (len(train_ex), len(eval_ex)) == (7, 3)
    assert report["source"] == str(data)
    holdout = {json.loads(line)["messages"][0]["content"] for line in (run_dir / "holdout.jsonl").read_text().splitlines()}
    train = {json.loads(line)["messages"][0]["content"] for line in (run_dir / "train.jsonl").read_text().splitlines()}
    assert not holdout & train
    assert len(starter._messages(rows[0], False)) == 1


def test_serverless_prepare_requires_a_file():
    with pytest.raises(SystemExit):
        starter._prepare(starter.parse_args(["prepare", "--run-id", "r"]))


def test_score_uses_json_keys_when_the_answer_is_json():
    ok, fields = starter._score('Sure: {"route": "big model", "math": true}', '{"route": "big model", "math": true}')
    assert ok and fields == {"route": True, "math": True}
    ok, fields = starter._score('{"route": "small model"}', '{"route": "big model", "math": true}')
    assert not ok and fields == {"route": False, "math": False}
    assert starter._score("  Paris ", "Paris") == (True, {})
    ok, fields = starter._score('{"route": "big model"}', '{"route": "big model", "note": null}')
    assert not ok and fields == {"route": True, "note": False}


def test_watch_does_not_count_queue_time_as_a_stall(monkeypatch):
    states = iter(["JOB_STATE_PENDING"] * 3 + ["JOB_STATE_COMPLETED"])
    clock = iter(range(0, 100_000, 3_600))

    def get(job_id):
        return SimpleNamespace(state=next(states), job_progress=SimpleNamespace(percent=0, epoch=0),
                               status=None, name="j", output_model="m", evaluation_dataset="",
                               metrics_file_signed_url=None, model_dump=lambda: {})

    monkeypatch.setattr(starter, "_connect",
                        lambda: (SimpleNamespace(supervised_fine_tuning_jobs=SimpleNamespace(get=get)), "acme"))
    monkeypatch.setattr(starter.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(starter.time, "sleep", lambda s: None)
    args = starter.parse_args(["watch", "--run-id", "r", "--no-progress-minutes", "30"])
    assert starter.cmd_watch(args) == 0


def test_parse_json_tolerates_prose():
    assert starter._parse_json('Sure: {"route": "big model"} done') == {"route": "big model"}
    assert starter._parse_json("no json") == {}


def test_missing_key_never_asks_for_chat(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        starter._connect()
    message = str(exc.value)
    assert "https://app.fireworks.ai/settings/users/api-keys" in message
    assert "Never paste it into chat" in message


def test_rejected_key_exits_with_the_server_message(monkeypatch):
    import fireworks
    import httpx

    def denied(self, **kwargs):
        response = httpx.Response(403, request=httpx.Request("GET", "https://api.fireworks.ai/v1/accounts"))
        raise fireworks.PermissionDeniedError(
            "denied", response=response, body={"code": 7, "message": "key not allowed from this network"}
        )

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test")
    monkeypatch.delenv("FIREWORKS_ACCOUNT_ID", raising=False)
    monkeypatch.setattr(fireworks.resources.accounts.AccountsResource, "list", denied)
    with pytest.raises(SystemExit) as exc:
        starter._connect()
    message = str(exc.value)
    assert "HTTP 403" in message
    assert "key not allowed from this network" in message
    assert "fw_test" not in message


def test_names_are_account_relative():
    names = starter._names("acme", "router-v1")
    assert names["job"] == "accounts/acme/supervisedFineTuningJobs/muse-starter-router-v1"
    assert names["output_model"] == "accounts/acme/models/muse-starter-router-v1"


def test_upload_validation_reports_bad_rows_without_dropping(tmp_path):
    f = tmp_path / "t.jsonl"
    good = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{}"}]}
    bad = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]}
    not_list = {"messages": "hi"}
    not_dict = {"messages": [{"role": "user", "content": "hi"}, "oops"]}
    f.write_text("\n".join(json.dumps(r) for r in (good, bad, good, not_list, not_dict)) + "\n")
    rows, problems = starter._validate_chat_jsonl(f)
    assert rows == 5
    assert problems == [
        "line 2: last message is not a non-empty assistant reply",
        "line 4: messages is not a non-empty list of objects",
        "line 5: messages is not a non-empty list of objects",
    ]


def test_render_summary_covers_every_row_and_reports_fetch_errors(monkeypatch):
    rows = [
        {"renderer": "muse_glimmer", "train_on_what": "LAST_ASSISTANT_MESSAGE", "token_ids": [1] * 50,
         "training_loss_weights": [0] * 40 + [1] * 10, "image_token_count": 600},
        {"renderer": "muse_glimmer", "train_on_what": "LAST_ASSISTANT_MESSAGE", "token_ids": [1] * 90,
         "training_loss_weights": [0] * 90, "image_token_count": 972},
    ]
    monkeypatch.setattr(starter, "_fetch_lines", lambda url: [json.dumps(r) for r in rows])
    summary = starter._render_summary("signed")
    assert summary["tokens_min_max"] == [50, 90]
    assert summary["rows_with_no_trained_tokens"] == 1
    assert summary["image_tokens_min_max"] == [600, 972]

    def not_ready(url):
        raise RuntimeError("404")

    monkeypatch.setattr(starter, "_fetch_lines", not_ready)
    assert "error" in starter._render_summary("signed")


def test_cleanup_reports_real_failures(monkeypatch, capsys):
    import fireworks
    import httpx

    def resp(code):
        return httpx.Response(code, request=httpx.Request("DELETE", "https://api.fireworks.ai/x"))

    def gone(*a, **k):
        raise fireworks.NotFoundError("not found", response=resp(404), body=None)

    def denied(*a, **k):
        raise fireworks.PermissionDeniedError("denied", response=resp(403), body=None)

    client = SimpleNamespace(models=SimpleNamespace(delete=gone),
                             supervised_fine_tuning_jobs=SimpleNamespace(delete=denied))
    monkeypatch.setattr(starter, "_connect", lambda: (client, "acme"))
    assert starter.cmd_cleanup(starter.parse_args(["cleanup", "--run-id", "r"])) == 1
    out = capsys.readouterr()
    assert "does not exist" in out.out
    assert "FAILED to delete job" in out.err
