#!/usr/bin/env python3
"""Muse Glimmer starter: a first SFT job on the preloaded prompt-router data.

Muse Glimmer 30B LoRA SFT is promotional ``$0`` on Managed SFT and on the Serverless
Training API for jobs started before ``2026-11-01T00:00:00-07:00``.
Source: https://docs.fireworks.ai/fine-tuning/models. RL, DPO, deployments,
inference, and Dedicated resources are not covered.

Managed SFT is the easiest path. It trains on the account's preloaded
``onboarding-prompt-router`` dataset in place and carves out an evaluation split::

    python -m examples.sft.muse_glimmer_starter plan
    python -m examples.sft.muse_glimmer_starter launch --surface managed   # prints a fresh run id
    python -m examples.sft.muse_glimmer_starter watch --run-id <id>

The Serverless Training API path is code-first and trains on your own chat JSONL.
It runs a short LoRA loop on the shared pool and compares base and tuned
snapshots on a held-out split through in-session sampling. ``prepare`` is local
and free: it validates the file, writes ``train.jsonl`` and ``holdout.jsonl`` to
the run directory, and prints one rendered example with its token count, so
inspect it before launching::

    python -m examples.sft.muse_glimmer_starter prepare --file data.jsonl  # prints a fresh run id
    python -m examples.sft.muse_glimmer_starter launch --surface serverless --file data.jsonl --run-id <same id>

``cleanup --run-id <id>`` deletes the managed job and its output model.
``plan`` is read-only. ``launch`` creates billable-surface resources and should
run only after the user approves the printed plan. Run from ``training/`` with
``FIREWORKS_API_KEY`` exported or set in a gitignored ``.env``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

# tinker imports pydantic.v1, which warns on Python 3.14; the warning is harmless here.
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"
RENDERER_NAME = "muse_glimmer"
# End of Oct 31 Pacific; Nov 1 00:00 is still PDT (UTC-7).
PROMOTION_DEADLINE = datetime(2026, 11, 1, tzinfo=timezone(timedelta(hours=-7)))
PROMOTION_SOURCE = "https://docs.fireworks.ai/fine-tuning/models"
LORA_ALPHA = 32
LORA_TARGETS = {"train_mlp": True, "train_attn": True, "train_unembed": True}
DEFAULT_DATASET = "onboarding-prompt-router"
TERMINAL = {"JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_EARLY_STOPPED"}


def _load_dotenv() -> None:
    try:
        import dotenv
    except ImportError:
        path = Path.cwd() / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True), override=False)


def _fetch_lines(url: str) -> list[str]:
    import certifi
    import httpx

    resp = httpx.get(url, timeout=30, verify=certifi.where())
    resp.raise_for_status()
    return resp.text.splitlines()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def promotion_active(now: datetime | None = None) -> bool:
    return (now or _now()) < PROMOTION_DEADLINE


def _error_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])
    return str(exc)


def _connect():
    """Return a client bound to the key's account, and that account ID."""
    from fireworks import AuthenticationError, Fireworks, PermissionDeniedError

    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit(
            "FIREWORKS_API_KEY is not set. Create a key at "
            "https://app.fireworks.ai/settings/users/api-keys, then run "
            "`read -s FIREWORKS_API_KEY && export FIREWORKS_API_KEY` in this terminal "
            "or add it to a gitignored .env. Never paste it into chat."
        )
    account = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not account:
        try:
            first = next(iter(Fireworks().accounts.list(page_size=1)), None)
        except (AuthenticationError, PermissionDeniedError) as exc:
            sys.exit(
                f"Fireworks rejected FIREWORKS_API_KEY (HTTP {exc.status_code}): {_error_message(exc)}\n"
                "Check that the key is current and belongs to the account you want, then retry. "
                "Nothing was created."
            )
        if first is None:
            sys.exit("Could not resolve the account for this API key; set FIREWORKS_ACCOUNT_ID.")
        account = first.name.split("/")[-1]
    return Fireworks(account_id=account, timeout=120), account


def _names(account: str, run_id: str) -> dict:
    job_id = f"muse-starter-{run_id}"
    serverless_id = f"muse-starter-sl-{run_id}"[:63].rstrip("-")
    return {
        "job_id": job_id,
        "job": f"accounts/{account}/supervisedFineTuningJobs/{job_id}",
        "output_model": f"accounts/{account}/models/{job_id}",
        "serverless_model_id": serverless_id,
        "serverless_model": f"accounts/{account}/models/{serverless_id}",
    }


def cmd_plan(args: argparse.Namespace) -> int:
    client, account = _connect()
    active = promotion_active()
    plan: dict = {
        "account": account,
        "utc_now": _now().isoformat(timespec="seconds"),
        "promotion": {
            "active": active,
            "deadline": PROMOTION_DEADLINE.isoformat(),
            "source": PROMOTION_SOURCE,
        },
        "base_model": BASE_MODEL,
        "method": "LoRA SFT",
        "run_id": args.run_id,
        "run_id_note": "pass --run-id to launch to use these resource names; IDs of deleted jobs must not be reused",
        "surface": args.surface,
    }
    if args.surface == "managed":
        try:
            ds = client.datasets.get(args.dataset)
            plan["dataset"] = {
                "name": ds.name,
                "state": ds.state,
                "rows": int(ds.example_count or 0),
                "format": ds.format,
            }
        except Exception as exc:  # noqa: BLE001 - report any lookup failure verbatim
            plan["dataset"] = {"name": f"accounts/{account}/datasets/{args.dataset}", "error": str(exc)[:200]}
        plan.update(
            {
                "evaluation": "eval_auto_carveout: the platform holds out a slice and reports eval loss; "
                "the slice size is chosen by the platform and not reported by the API",
                "parameters": {"lora_rank": args.lora_rank, "epochs": args.epochs, "learning_rate": args.learning_rate,
                               "max_context_length": args.max_context_length},
                "resources": _names(account, args.run_id),
            }
        )
    else:
        plan.update(
            {
                "dataset": {"file": args.file or "required: pass --file <chat.jsonl>",
                            "holdout_rows": args.n_eval, "train_rows": args.n_train or "all remaining rows"},
                "evaluation": "exact match (and per-key accuracy for JSON answers), base vs tuned, same holdout, "
                              "in-session sampling",
                "parameters": {"lora_rank": args.lora_rank, "epochs": args.epochs, "batch_size": args.batch_size,
                               "learning_rate": args.learning_rate, "max_seq_len": args.max_seq_len,
                               "max_eval_tokens": args.max_eval_tokens, "seed": args.seed,
                               "lora_alpha": LORA_ALPHA, "lora_targets": LORA_TARGETS,
                               "train_on": "last_assistant_turn", "loss": "cross_entropy",
                               "optimizer": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
                                             "weight_decay": 0.0, "grad_clip_norm": 1.0},
                               "emit_grad_norm_metrics": True},
                "resources": {"promoted_model": None if args.no_promote else _names(account, args.run_id)["serverless_model"],
                              "note": "the tuned sampler checkpoint is promoted to a LoRA model before the session "
                                      "closes; session checkpoints are unreachable after the session ends"},
            }
        )
    plan["cost"] = {
        "training": "$0 (promotional)" if active else "paid: resolve the live rate at https://fireworks.ai/pricing",
        "excluded": "full-parameter SFT, RL, DPO, Dedicated, deployment, inference outside the run; in-session "
        "evaluation sampling is a separate line unless the account shows it covered",
    }
    print(json.dumps(plan, indent=2))
    return 0


def cmd_launch_managed(args: argparse.Namespace) -> int:
    client, account = _connect()
    names = _names(account, args.run_id)
    if not promotion_active() and not args.allow_paid:
        sys.exit("The Muse Glimmer promotion has ended. Re-run with --allow-paid after approving the live rate.")
    try:
        existing = client.supervised_fine_tuning_jobs.get(names["job_id"])
    except Exception:  # noqa: BLE001 - not found is the expected case
        existing = None
    if existing is not None:
        wanted = {"base_model": BASE_MODEL, "dataset": f"accounts/{account}/datasets/{args.dataset}",
                  "output_model": names["output_model"], "epochs": args.epochs, "lora_rank": args.lora_rank,
                  "learning_rate": args.learning_rate, "max_context_length": args.max_context_length}
        def same(have, want) -> bool:
            if isinstance(want, float) and isinstance(have, (int, float)):
                return abs(have - want) <= 1e-9 + 1e-6 * abs(want)
            return have == want

        mismatched = {k: {"existing": getattr(existing, k, None), "requested": v}
                      for k, v in wanted.items() if not same(getattr(existing, k, None), v)}
        print(json.dumps({"reused": not mismatched, "job": existing.name, "state": existing.state,
                          "mismatched": mismatched}, indent=2, default=str))
        if mismatched:
            print("This run ID belongs to a job with different settings; pass a new --run-id.", file=sys.stderr)
        return 0 if not mismatched else 1
    job = client.supervised_fine_tuning_jobs.create(
        supervised_fine_tuning_job_id=names["job_id"],
        dataset=f"accounts/{account}/datasets/{args.dataset}",
        base_model=BASE_MODEL,
        output_model=names["output_model"],
        display_name=f"Muse Glimmer starter {args.run_id}",
        epochs=args.epochs,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        max_context_length=args.max_context_length,
        eval_auto_carveout=True,
    )
    print(json.dumps({"created": job.name, "state": job.state, "output_model": names["output_model"],
                      "run_id": args.run_id}, indent=2))
    return 0


def _metrics_summary(signed_url: str | None) -> dict:
    """Losses plus the resolved learning rate, steps, and tokens from metrics.jsonl."""
    if not signed_url:
        return {"error": "metrics file not available yet"}
    try:
        lines = _fetch_lines(signed_url)
    except Exception as exc:  # noqa: BLE001 - report instead of hiding
        return {"error": f"could not read metrics file: {str(exc)[:160]}"}
    series: dict = {}
    resolved: dict = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for key, value in row.items():
            if "loss" in key and isinstance(value, (int, float)):
                series.setdefault(key, []).append(value)
        for key in ("train/lr", "train/step", "train/total_tokens"):
            if key in row:
                resolved[key] = row[key]
    out = {k: {"first": v[0], "last": v[-1], "points": len(v)} for k, v in series.items()}
    out["resolved"] = resolved
    return out


def _render_summary(signed_url: str | None, rows: int = 3) -> dict:
    """What the platform actually rendered for training: masking, image tokens, one decoded row."""
    if not signed_url:
        return {}
    try:
        samples = [json.loads(line) for line in _fetch_lines(signed_url) if line.strip()]
    except Exception as exc:  # noqa: BLE001
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return {"error": f"could not read render samples ({status or type(exc).__name__})"}
    if not samples:
        return {}
    first = samples[0]

    def span(values: list[int]) -> list[int]:
        return [min(values), max(values)]

    trained = [sum(1 for w in (s.get("training_loss_weights") or s.get("token_weights") or []) if w)
               for s in samples]
    return {
        "samples": len(samples),
        "renderer": first.get("renderer"),
        "train_on_what": sorted({str(s.get("train_on_what")) for s in samples}),
        "tokens_min_max": span([len(s.get("token_ids") or s.get("decoded_tokens") or []) for s in samples]),
        "trained_tokens_min_max": span(trained),
        "rows_with_no_trained_tokens": sum(1 for t in trained if not t),
        "image_tokens_min_max": span([int(s.get("image_token_count") or 0) for s in samples]),
        "first_row_decoded_head": "".join(first.get("decoded_tokens") or [])[:400] if rows else "",
    }


def cmd_watch(args: argparse.Namespace) -> int:
    if not (args.run_id or args.job_id):
        sys.exit("Pass --run-id (printed by launch) or --job-id.")
    client, account = _connect()
    job_id = args.job_id or _names(account, args.run_id)["job_id"]
    last_progress, last_change = None, time.monotonic()
    rendered = False
    while True:
        job = client.supervised_fine_tuning_jobs.get(job_id)
        progress = job.job_progress
        percent = getattr(progress, "percent", None) if progress else None
        epoch = getattr(progress, "epoch", None) if progress else None
        message = getattr(job.status, "message", "") if job.status else ""
        if message and "%" in message:  # the status message already carries percent and epoch
            detail = message
        else:
            epoch_part = f" epoch={epoch}" if epoch else ""  # the API can report 0 even at 100%
            detail = f"progress={percent}%{epoch_part} {message}".rstrip()
        print(f"{_now().isoformat(timespec='seconds')} {job.state} {detail}", flush=True)
        urls = job.model_dump()
        if not rendered and urls.get("renderSamplesSignedUrl"):
            summary = _render_summary(urls["renderSamplesSignedUrl"])
            if summary.get("error"):
                print(f"render samples not ready yet; will retry ({summary['error']})", flush=True)
            elif summary:
                print(json.dumps({"render_samples": summary}, indent=2), flush=True)
                rendered = True
        if job.state in TERMINAL:
            report = {"job": job.name, "state": job.state, "output_model": job.output_model,
                      "eval_split": (f"dataset {job.evaluation_dataset}" if job.evaluation_dataset else
                                     "auto carve-out; size chosen by the platform, not reported by the API"),
                      "metrics": _metrics_summary(job.metrics_file_signed_url)}
            if job.status and getattr(job.status, "message", None):
                report["message"] = job.status.message
            print(json.dumps(report, indent=2))
            return 0 if job.state == "JOB_STATE_COMPLETED" else 1
        if percent != last_progress or job.state != "JOB_STATE_RUNNING":
            last_progress, last_change = percent, time.monotonic()
        elif time.monotonic() - last_change > args.no_progress_minutes * 60:
            print(f"No progress for {args.no_progress_minutes} minutes; route to debug with this job ID.")
            return 2
        if args.once:
            return 0
        time.sleep(args.interval)


def _validate_chat_jsonl(path: Path) -> tuple[int, list[str]]:
    """Return (row count, problems). Rows are never dropped or rewritten here."""
    problems, rows = [], 0
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            msgs = json.loads(line)["messages"]
        except (ValueError, KeyError, TypeError):
            problems.append(f"line {n}: not a JSON object with messages")
            continue
        if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
            problems.append(f"line {n}: messages is not a non-empty list of objects")
            continue
        if msgs[-1].get("role") != "assistant" or not str(msgs[-1].get("content") or "").strip():
            problems.append(f"line {n}: last message is not a non-empty assistant reply")
    return rows, problems


def cmd_upload(args: argparse.Namespace) -> int:
    if not args.file or args.dataset == DEFAULT_DATASET:
        sys.exit("Pass --file <chat.jsonl> and a new --dataset <id> for your data.")
    path = Path(args.file)
    rows, problems = _validate_chat_jsonl(path)
    if problems:
        print(json.dumps({"rows": rows, "problems": problems[:50]}, indent=2))
        sys.exit("Fix or remove these rows in a copy of the file (with the user's approval), then upload the copy.")
    client, account = _connect()
    try:
        ds = client.datasets.get(args.dataset)
    except Exception:  # noqa: BLE001 - not found is the expected case
        ds = None
    if ds is not None and ds.state == "READY":
        sys.exit(f"Dataset {ds.name} already exists with {ds.example_count} rows; "
                 "upload to a new --dataset ID instead of reusing it.")
    if ds is None:
        ds = client.datasets.create(dataset_id=args.dataset,
                                    dataset={"displayName": args.dataset, "exampleCount": str(rows), "format": "CHAT"})
    if ds.state != "READY":
        with path.open("rb") as f:
            client.datasets.upload(args.dataset, file=f)
        for _ in range(60):
            ds = client.datasets.get(args.dataset)
            if ds.state == "READY":
                break
            time.sleep(5)
    print(json.dumps({"dataset": ds.name, "state": ds.state, "rows": ds.example_count}, indent=2))
    return 0 if ds.state == "READY" else 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    if not args.run_id:
        sys.exit("Pass the --run-id printed by launch.")
    import fireworks

    client, account = _connect()
    names = _names(account, args.run_id)
    failed = False
    for label, delete in (
        ("output_model", lambda: client.models.delete(names["job_id"])),
        ("job", lambda: client.supervised_fine_tuning_jobs.delete(names["job_id"])),
        ("serverless_model", lambda: client.models.delete(names["serverless_model_id"])),
    ):
        try:
            delete()
            print(f"deleted {label} {names[label]}")
        except fireworks.NotFoundError:
            print(f"skip {label}: {names[label]} does not exist")
        except fireworks.APIStatusError as exc:
            failed = True
            print(f"FAILED to delete {label} {names[label]}: HTTP {exc.status_code} {exc.message}", file=sys.stderr)
    return 1 if failed else 0


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _messages(ex: dict, with_answer: bool) -> list[dict]:
    msgs = ex["messages"]
    return msgs if with_answer else msgs[:-1]


def _score(pred_text: str, gold_text: str) -> tuple[bool, dict[str, bool]]:
    """Exact match for one answer, plus per-key matches when the reference answer is a JSON object."""
    gold = _parse_json(gold_text)
    if not gold:
        return " ".join(pred_text.split()) == " ".join(gold_text.split()), {}
    pred = _parse_json(pred_text)
    fields = {k: k in pred and str(pred[k]).strip().lower() == str(v).strip().lower() for k, v in gold.items()}
    return all(fields.values()), fields


def _prepare(args: argparse.Namespace):
    """Split the user's chat JSONL into train and holdout in the run directory and render one example."""
    from training.renderer import TrainOnWhat, get_renderer
    from training.renderer.tokenizer import get_tokenizer

    if not args.file:
        sys.exit("Serverless Training runs on your own chat JSONL: pass --file <chat.jsonl>.")
    path = Path(args.file)
    count, problems = _validate_chat_jsonl(path)
    if problems:
        print(json.dumps({"rows": count, "problems": problems[:50]}, indent=2))
        sys.exit("Fix or remove these rows in a copy of the file (with the user's approval), then rerun.")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) <= args.n_eval:
        sys.exit(f"{len(rows)} rows is too few to hold out {args.n_eval}; lower --n-eval or add rows.")
    random.Random(args.seed).shuffle(rows)
    eval_ex = rows[: args.n_eval]
    train_ex = rows[args.n_eval :][: args.n_train] if args.n_train else rows[args.n_eval :]

    run_dir = Path(args.log_path or f"./muse_starter_logs/{args.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train_ex), ("holdout.jsonl", eval_ex)):
        with (run_dir / name).open("w") as f:
            for ex in split:
                f.write(json.dumps({"messages": _messages(ex, True)}) + "\n")

    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    renderer = get_renderer(RENDERER_NAME, tokenizer)
    model_input, weights = renderer.build_supervised_example(
        _messages(train_ex[0], True), train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    tokens = model_input.to_ints()
    report = {
        "run_dir": str(run_dir),
        "source": str(path),
        "train_rows": len(train_ex),
        "holdout_rows": len(eval_ex),
        "example_tokens": len(tokens),
        "example_trained_tokens": int(sum(1 for w in weights.tolist() if w > 0)),
        "max_seq_len": args.max_seq_len,
        "fits": len(tokens) <= args.max_seq_len,
        "example_rendered": tokenizer.decode(tokens),
    }
    return run_dir, train_ex, eval_ex, tokenizer, renderer, report


def _mcnemar(base: list[bool], tuned: list[bool]) -> dict:
    """Exact two-sided McNemar test on per-example exact match."""
    from math import comb

    only_tuned = sum(1 for b, t in zip(base, tuned) if t and not b)
    only_base = sum(1 for b, t in zip(base, tuned) if b and not t)
    n = only_tuned + only_base
    k = min(only_tuned, only_base)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_tuned_correct": only_tuned, "only_base_correct": only_base, "p_value": p}


def cmd_prepare(args: argparse.Namespace) -> int:
    report = _prepare(args)[-1]
    print(json.dumps(report, indent=2))
    return 0 if report["fits"] else 1


def cmd_launch_serverless(args: argparse.Namespace) -> int:
    if not promotion_active() and not args.allow_paid:
        sys.exit("The Muse Glimmer promotion has ended. Re-run with --allow-paid after approving the live rate.")
    _, account = _connect()
    import tinker
    from fireworks.training.sdk import FiretitanServiceClient

    from training.renderer import TrainOnWhat, get_text_content
    from training.utils.serverless import enable_serverless_supervised_409_retry, serverless_base_url

    run_dir, train_ex, eval_ex, tokenizer, renderer, _ = _prepare(args)
    messages = _messages

    def datum(ex: dict):
        model_input, weights = renderer.build_supervised_example(
            messages(ex, True), train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
        )
        tokens = [int(t) for t in model_input.to_ints()]
        w = [float(x) for x in weights.tolist()]
        if len(tokens) > args.max_seq_len:
            return "oversized"
        if len(tokens) < 2 or not any(x > 0 for x in w[1:]):
            return None
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"target_tokens": tokens[1:], "weights": w[1:]},
        )

    rendered = [datum(ex) for ex in train_ex]
    oversized = [i for i, d in enumerate(rendered) if d == "oversized"]
    if oversized:
        sys.exit(f"{len(oversized)} train rows exceed --max-seq-len {args.max_seq_len} (row indices "
                 f"{oversized[:20]}); raise --max-seq-len or remove them before launch.")
    datums = [d for d in rendered if d is not None]
    sample = renderer.build_generation_prompt(messages(train_ex[0], False))
    print(f"rendered {len(datums)}/{len(train_ex)} train rows; holdout {len(eval_ex)} rows; "
          f"example prompt tokens={len(sample.to_ints())}", flush=True)

    service = FiretitanServiceClient(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=serverless_base_url(os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")),
    )
    enable_serverless_supervised_409_retry(service)
    log = (run_dir / "metrics.jsonl").open("a")
    try:
        training_client = service.create_lora_training_client(base_model=BASE_MODEL, rank=args.lora_rank,
                                                              alpha=LORA_ALPHA, **LORA_TARGETS)
        print(f"session={getattr(service, 'training_session_id', None)} run={getattr(training_client, 'run_id', None)}",
              flush=True)
        params = tinker.SamplingParams(max_tokens=args.max_eval_tokens, temperature=0.0,
                                       stop=renderer.get_stop_sequences())

        per_example: dict[str, list[bool]] = {}

        def evaluate(label: str) -> dict:
            snapshot = training_client.save_weights_for_sampler(f"starter-{label}").result().path
            sampler = service.create_sampling_client(model_path=snapshot, tokenizer=tokenizer)
            hits: dict[str, int] = {}
            seen: dict[str, int] = {}
            exact = 0
            correct: list[bool] = []
            hit_limit = 0
            preds = (run_dir / f"predictions_{label}.jsonl").open("w")
            prompt_tokens = sampled_tokens = 0
            try:
                prompts = [renderer.build_generation_prompt(messages(ex, False)) for ex in eval_ex]
                futures = [sampler.sample(prompt=pr, num_samples=1, sampling_params=params) for pr in prompts]
                for i, (ex, prompt, fut) in enumerate(zip(eval_ex, prompts, futures)):
                    res = fut.result() if hasattr(fut, "result") else fut
                    out_tokens = list(res.sequences[0].tokens)
                    prompt_tokens += len(prompt.to_ints())
                    sampled_tokens += len(out_tokens)
                    if (i + 1) % 10 == 0 or i + 1 == len(eval_ex):
                        print(f"{label}: scored {i + 1}/{len(eval_ex)}", flush=True)
                    msg, termination = renderer.parse_response(out_tokens)
                    text = get_text_content(msg)
                    truncated = not termination.is_clean
                    hit_limit += truncated
                    gold = str(ex["messages"][-1].get("content") or "")
                    ok, fields = _score(text, gold)
                    exact += ok
                    correct.append(ok)
                    preds.write(json.dumps({"index": i, "gold": gold[:2000], "exact": ok, "fields": fields,
                                            "truncated": truncated, "answer_text": text[:2000]}) + "\n")
                    for f, hit in fields.items():
                        seen[f] = seen.get(f, 0) + 1
                        hits[f] = hits.get(f, 0) + hit
            finally:
                preds.close()
                close = getattr(sampler, "close", None)
                if close:
                    close()
            per_example[label] = correct
            n = len(eval_ex)
            result = {"label": label, "exact_match": exact / n, "unterminated": hit_limit,
                      "prompt_tokens": prompt_tokens, "sampled_tokens": sampled_tokens,
                      "field_accuracy": {f: hits[f] / seen[f] for f in seen}}
            log.write(json.dumps({"eval": result}) + "\n")
            print(json.dumps(result), flush=True)
            return result

        base = evaluate("base")
        adam = tinker.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, eps=1e-8,
                                 weight_decay=0.0, grad_clip_norm=1.0)
        rng = random.Random(args.seed)
        step = 0
        for epoch in range(args.epochs):
            order = list(range(len(datums)))
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                batch = [datums[i] for i in order[start : start + args.batch_size]]
                fb = training_client.forward_backward(batch, "cross_entropy").result()
                opt = training_client.optim_step(adam, emit_grad_norm_metrics=True).result()
                step += 1
                metrics = dict(getattr(fb, "metrics", None) or {})
                tokens = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or 1.0
                loss = metrics.get("loss:sum", 0.0) / max(float(tokens), 1.0)
                grad = {k: v for k, v in (getattr(opt, "metrics", None) or {}).items() if "grad" in k}
                row = {"step": step, "epoch": epoch + 1, "loss": round(loss, 4), **grad}
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        tuned = evaluate("tuned")
        promoted = None if args.no_promote else _promote_tuned(service, account, args.run_id)
        summary = {"steps": step, "base": base, "tuned": tuned, "promoted_model": promoted,
                   "exact_match_delta": round(tuned["exact_match"] - base["exact_match"], 4),
                   "paired": _mcnemar(per_example["base"], per_example["tuned"]),
                   "session": getattr(service, "training_session_id", None), "log": str(run_dir)}
        prompt_tokens = base["prompt_tokens"] + tuned["prompt_tokens"]
        sampled_tokens = base["sampled_tokens"] + tuned["sampled_tokens"]
        summary["cost"] = {"training": "$0 (promotional)" if promotion_active() else "paid at the live rate",
                           "evaluation_sampling": {"prompt_tokens": prompt_tokens, "sampled_tokens": sampled_tokens,
                                                   "note": "billed separately from training at published "
                                                           "per-token rates"}}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"cost: training {summary['cost']['training']}; evaluation sampling used {prompt_tokens} prompt + "
              f"{sampled_tokens} sampled tokens, billed separately at published per-token rates", flush=True)
    finally:
        log.close()
        service.close()
        print("Serverless session closed; no deployment was created. Its training slot frees when the "
              "session expires, so a new run started right away may wait or get a 429.", flush=True)
    return 0


def _promote_tuned(service, account: str, run_id: str) -> str | None:
    """Promote the tuned sampler checkpoint to a LoRA model before the session (and its checkpoints) go away."""
    from fireworks.training.sdk import FireworksClient

    fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"],
                         base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
    rows = [r for r in fw.list_training_session_checkpoints(service.training_session_name)
            if r.get("promotable") and "starter-tuned" in (r.get("checkpointName") or r.get("name") or "")]
    if not rows:
        print("no promotable tuned checkpoint found; the adapter stays session-only", flush=True)
        return None
    target = max(rows, key=lambda r: r.get("createTime", ""))
    model_id = _names(account, run_id)["serverless_model_id"]
    fw.promote_session_checkpoint(name=target["name"], output_model_id=model_id, base_model=BASE_MODEL)
    name = f"accounts/{account}/models/{model_id}"
    print(json.dumps({"promoted_model": name, "from_checkpoint": target["name"]}), flush=True)
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare", "upload", "launch", "watch", "cleanup"):
        s = sub.add_parser(name)
        s.add_argument("--surface", choices=("managed", "serverless"), default="managed")
        s.add_argument("--run-id", default=None,
                       help="Names the job, models, and log dir. plan/prepare/launch generate a fresh one when omitted; "
                            "never reuse the ID of a deleted job.")
        s.add_argument("--dataset", default=DEFAULT_DATASET, help="Account-relative dataset ID (managed).")
        s.add_argument("--file", default=None,
                       help="Local chat JSONL: uploaded as --dataset by upload; trained on by the serverless path.")
        s.add_argument("--lora-rank", type=int, default=8)
        s.add_argument("--epochs", type=int, default=1)
        s.add_argument("--learning-rate", type=float, default=1e-4)
        s.add_argument("--max-context-length", type=int, default=4096)
        s.add_argument("--batch-size", type=int, default=16)
        s.add_argument("--max-seq-len", type=int, default=2048)
        s.add_argument("--max-eval-tokens", type=int, default=2048)
        s.add_argument("--n-train", type=int, default=None, help="Serverless: cap on train rows (default: all).")
        s.add_argument("--n-eval", type=int, default=40, help="Serverless: rows held out for base vs tuned.")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--log-path", default=None)
        s.add_argument("--allow-paid", action="store_true", help="Launch after the promotion ends.")
        s.add_argument("--no-promote", action="store_true",
                       help="Serverless: keep the tuned adapter session-only instead of promoting it to a model.")
        s.add_argument("--job-id", default=None, help="Watch an existing managed job.")
        s.add_argument("--interval", type=int, default=60)
        s.add_argument("--no-progress-minutes", type=int, default=30)
        s.add_argument("--once", action="store_true", help="Poll once and exit.")
    args = p.parse_args(argv)
    if args.run_id is None and args.command in ("plan", "prepare", "launch"):
        args.run_id = f"r{_now():%Y%m%d-%H%M%S}"
        serverless_without_file = (args.command == "prepare" or args.surface == "serverless") and not args.file
        if args.command != "plan" and not serverless_without_file:
            print(f"run id: {args.run_id} (pass --run-id {args.run_id} to watch and cleanup)", file=sys.stderr)
    return args


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = parse_args(argv)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "upload":
        return cmd_upload(args)
    if args.command == "watch":
        return cmd_watch(args)
    if args.command == "cleanup":
        return cmd_cleanup(args)
    if args.surface == "managed":
        return cmd_launch_managed(args)
    return cmd_launch_serverless(args)


if __name__ == "__main__":
    sys.exit(main())
