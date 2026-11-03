#!/usr/bin/env python3
"""Ask Before Acting: multi-turn RL on Fireworks serverless training.

One file, one loop, modeled on ``examples/archive/countdown/countdown_rl.py`` and
the tool-calling episode of ``case-studies/agentic_rl_text2sql``. Each episode
is a short conversation: the user asks for something that may be missing a
detail, the policy may call ``ask_user`` (a coded simulator answers from the
hidden task spec), and the episode ends when the policy calls an action tool.
``aba_env.score_episode`` turns the transcript into one reward.

Every assistant turn becomes its own importance-sampling datum carrying the
episode's group-relative advantage (GRPO over ``group_size`` episodes of the
same task), so a question that led to a correct action is reinforced together
with the action.

Usage:
    export FIREWORKS_API_KEY=fw_...
    python ask_before_acting_rl.py --prepare-dataset
    python ask_before_acting_rl.py --steps 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
from fireworks.training.sdk import (
    FireworksClient,
    FiretitanSamplingParams,
    FiretitanServiceClient,
)
from fireworks.training.sdk.routing import copy_routing, routing_model_input_kwargs

import training.renderer  # noqa: F401  (registers renderers)
from training.renderer import get_renderer
from training.utils import resolve_router_replay_enabled
from training.utils.rl.router_replay import build_r3_routing_matrices, validate_r3_routing_matrices
from training.utils.supervised import resolve_renderer_name
from training.utils.tokenizers import load_tokenizer

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aba_env import (  # noqa: E402
    SYSTEM_PROMPT,
    generate_tasks,
    score_episode,
    simulated_user_reply,
    task_to_row,
    tool_specs,
)

DEFAULT_DATASET = HERE / "data" / "aba_tasks.jsonl"
MAX_PROMOTABLE_CHECKPOINT_NAME_LEN = 17


@dataclass
class Config:
    base_model: str = "accounts/fireworks/models/glm-5p3-flash"
    tokenizer_model: str = "zai-org/GLM-5.3-Flash"
    renderer_name: str = ""
    lora_rank: int = 64
    lora_alpha: int = 128
    max_seq_len: int = 32768

    dataset: str = str(DEFAULT_DATASET)
    seed: int = 0
    eval_prompt_groups: int = 16
    eval_seed: int = 1

    steps: int = 20
    prompt_groups_per_step: int = 8
    group_size: int = 8
    episode_concurrency: int = 16
    max_turns: int = 4
    max_sample_tokens: int = 1024
    temperature: float = 1.0
    # Evaluation samples `eval_group_size` episodes per held-out task at the
    # training temperature and reports, besides the mean, the share of tasks
    # the policy gets right EVERY time (pass^k). Greedy decoding is nearly
    # perfect on this environment; consistency under sampling is what trains.
    eval_temperature: float = 1.0
    eval_group_size: int = 4
    learning_rate: float = 2e-5
    router_replay: bool = True
    eval_interval: int = 5

    checkpoint_name: str = "aba-sample"
    final_checkpoint_name: str = "aba-final"
    output_model_id: str = ""
    sampling_timeout_s: float = 1800.0
    run_dir: str = ""
    api_base_url: str = field(default_factory=lambda: os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
    api_key: str = field(default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", ""))


def _serverless_base_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/training/v1/serverless"):
        return root
    if root.endswith("/training/v1"):
        return f"{root}/serverless"
    return f"{root}/training/v1/serverless"


def _control_plane_base_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    suffix = "/training/v1/serverless"
    return root[: -len(suffix)] if root.endswith(suffix) else root


def prepare_dataset(output: Path, num_rows: int, seed: int) -> None:
    tasks = generate_tasks(num_rows, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for t in tasks:
            f.write(json.dumps(task_to_row(t)) + "\n")
    n_missing = [len(t["missing"]) for t in tasks]
    n_vague = sum(bool(t["vague"]) for t in tasks)
    print(f"wrote {len(tasks)} tasks -> {output}  (unknown fields per task: "
          f"0:{n_missing.count(0)} 1:{n_missing.count(1)} 2:{n_missing.count(2)} 3+:{sum(n >= 3 for n in n_missing)}; "
          f"{n_vague} tasks contain a vague detail)")


def _group_relative_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    if len(rewards) <= 1:
        return [0.0 for _ in rewards]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    std = math.sqrt(var)
    if std < eps:
        return [0.0 for _ in rewards]
    return [(r - mean) / (std + eps) for r in rewards]


def _tool_call_args(tc: Any) -> dict[str, Any]:
    raw = getattr(getattr(tc, "function", None), "arguments", None)
    if raw is None and isinstance(tc, dict):
        raw = (tc.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _tool_call_name(tc: Any) -> str:
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    if isinstance(tc, dict):
        return str((tc.get("function") or {}).get("name") or "")
    return ""


def _tool_call_id(tc: Any) -> str:
    return str(getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "") or "")


class ServerlessAskBeforeActingRL:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        rows = [json.loads(l) for l in Path(cfg.dataset).open() if l.strip()]
        order = list(range(len(rows)))
        random.Random(cfg.eval_seed).shuffle(order)
        eval_idx = set(order[: cfg.eval_prompt_groups])
        self.eval_rows = [rows[i] for i in sorted(eval_idx)]
        self.train_rows = [r for i, r in enumerate(rows) if i not in eval_idx]
        random.Random(cfg.seed).shuffle(self.train_rows)
        self.cursor = 0

        self.tokenizer = load_tokenizer(cfg.tokenizer_model)
        self.renderer_name = resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name)
        self.renderer = get_renderer(self.renderer_name, self.tokenizer)
        self.tools = tool_specs()
        self.prefix = self.renderer.create_conversation_prefix_with_tools(self.tools, SYSTEM_PROMPT)

        self.service = FiretitanServiceClient(api_key=cfg.api_key, base_url=_serverless_base_url(cfg.api_base_url))
        self.training_client = self.service.create_lora_training_client(
            base_model=cfg.base_model, rank=cfg.lora_rank, alpha=cfg.lora_alpha
        )
        self.router_replay_enabled = resolve_router_replay_enabled(
            requested=cfg.router_replay,
            api_key=cfg.api_key,
            base_url=_control_plane_base_url(cfg.api_base_url),
            additional_headers=None,
            base_model=cfg.base_model,
            training_client=self.training_client,
        )

        self.run_dir = Path(cfg.run_dir).resolve() if cfg.run_dir else Path("/tmp") / f"aba_rl_{int(time.time())}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "eval_transcripts").mkdir(exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.eval_metrics_path = self.run_dir / "eval_metrics.jsonl"
        print(
            f"connected serverless session={getattr(self.service, 'training_session_name', None)} "
            f"run={getattr(self.training_client, 'run_id', None)}\n"
            f"base_model={cfg.base_model} renderer={self.renderer_name} lora_rank={cfg.lora_rank} "
            f"alpha={cfg.lora_alpha} lr={cfg.learning_rate} router_replay={self.router_replay_enabled}\n"
            f"train_rows={len(self.train_rows)} eval_rows={len(self.eval_rows)} steps={cfg.steps} "
            f"prompt_groups_per_step={cfg.prompt_groups_per_step} group_size={cfg.group_size} "
            f"max_turns={cfg.max_turns}\nrun_dir={self.run_dir}",
            flush=True,
        )

    # ----- one episode -------------------------------------------------------
    def run_episode(self, sampler: Any, row: dict[str, Any], temperature: float) -> dict[str, Any]:
        cfg = self.cfg
        task = json.loads(row["ground_truth"])
        messages = list(self.prefix) + [{"role": "user", "content": row["messages"][-1]["content"]}]
        params = FiretitanSamplingParams(
            max_tokens=cfg.max_sample_tokens,
            temperature=temperature,
            stop=self.renderer.get_stop_sequences(),
            include_routing_matrix=self.router_replay_enabled,
        )
        turns: list[dict[str, Any]] = []
        revealed: set[str] = set()
        unnecessary = 0
        questions = 0
        action_call = None
        ended = "max_turns"
        for _ in range(cfg.max_turns):
            prompt = self.renderer.build_generation_prompt(messages)
            if prompt.length + cfg.max_sample_tokens > cfg.max_seq_len:
                ended = "context"
                break
            result = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result(timeout=cfg.sampling_timeout_s)
            seqs = getattr(result, "sequences", []) or []
            if not seqs:
                ended = "no_sample"
                break
            seq = seqs[0]
            tokens = list(getattr(seq, "tokens", []) or [])
            logprobs = getattr(seq, "logprobs", None)
            if not tokens or logprobs is None or len(logprobs) != len(tokens):
                ended = "bad_sample"
                break
            turns.append({"prompt": prompt, "tokens": tokens, "logprobs": [float(x) for x in logprobs],
                          "routing": copy_routing(getattr(seq, "routing_matrices", None))})
            msg, _ok = self.renderer.parse_response(tokens)
            messages.append(msg)
            tool_calls = list(msg.get("tool_calls") or [])
            if not tool_calls:
                ended = "no_tool_call"   # answered in prose instead of acting
                break
            tc = tool_calls[0]
            name, args = _tool_call_name(tc), _tool_call_args(tc)
            if name == "ask_user":
                questions += 1
                reply, newly, unneeded = simulated_user_reply(str(args.get("question", "")), task, revealed)
                revealed.update(newly)
                unnecessary += int(unneeded)
                messages.append({"role": "tool", "content": reply, "tool_call_id": _tool_call_id(tc), "name": name})
                continue
            action_call = {"name": name, "arguments": args}
            ended = "acted"
            break
        score = score_episode(task, action_call, unnecessary, questions)
        return {"turns": turns, "score": score, "questions": questions, "ended": ended,
                "task": task, "transcript": _transcript(messages[len(self.prefix):])}

    # ----- evaluation --------------------------------------------------------
    def evaluate(self, snapshot: str, completed_steps: int) -> dict[str, Any]:
        cfg = self.cfg
        t0 = time.time()
        sampler = self.service.create_sampling_client(model_path=snapshot, tokenizer=self.tokenizer)
        jobs = [(ti, row) for ti, row in enumerate(self.eval_rows) for _ in range(cfg.eval_group_size)]
        try:
            with ThreadPoolExecutor(cfg.episode_concurrency) as ex:
                tagged = list(ex.map(lambda j: (j[0], self.run_episode(sampler, j[1], cfg.eval_temperature)), jobs))
        finally:
            sampler.close()
        episodes = [ep for _, ep in tagged]
        by_task: dict[int, list[bool]] = {}
        for ti, ep in tagged:
            by_task.setdefault(ti, []).append(bool(ep["score"]["right_args"]))
        n_tasks = max(1, len(by_task))
        rec = {"completed_steps": completed_steps, "snapshot": snapshot, **_summarize("eval", episodes),
               "eval/tasks": len(by_task), "eval/samples_per_task": cfg.eval_group_size,
               "eval/task_all_correct_rate": sum(all(v) for v in by_task.values()) / n_tasks,
               "eval/task_any_correct_rate": sum(any(v) for v in by_task.values()) / n_tasks,
               "perf/eval_wall_time": time.time() - t0}
        with self.eval_metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        with (self.run_dir / "eval_transcripts" / f"step-{completed_steps:04d}.jsonl").open("w") as f:
            for _ti, ep in tagged:
                f.write(json.dumps({"task": ep["task"], "score": ep["score"], "ended": ep["ended"],
                                    "questions": ep["questions"], "transcript": ep["transcript"]}, ensure_ascii=False) + "\n")
        print(
            f"eval after_steps={completed_steps:02d} reward={rec['eval/reward']:.3f} "
            f"correct={rec['eval/correct_rate']:.1%} all_correct={rec['eval/task_all_correct_rate']:.1%} "
            f"acted={rec['eval/act_rate']:.1%} "
            f"questions={rec['eval/mean_questions']:.2f} unneeded={rec['eval/mean_unnecessary_questions']:.2f} "
            f"elapsed={rec['perf/eval_wall_time']:.1f}s",
            flush=True,
        )
        return rec

    # ----- one optimizer step -----------------------------------------------
    def step(self, step: int) -> dict[str, Any]:
        cfg = self.cfg
        t0 = time.time()
        snapshot = self.training_client.save_weights_for_sampler(f"{cfg.checkpoint_name}-{step:04d}").result().path
        batch = [self.train_rows[(self.cursor + i) % len(self.train_rows)] for i in range(cfg.prompt_groups_per_step)]
        self.cursor += cfg.prompt_groups_per_step

        sampler = self.service.create_sampling_client(model_path=snapshot, tokenizer=self.tokenizer)
        try:
            jobs = [(gi, row) for gi, row in enumerate(batch) for _ in range(cfg.group_size)]
            with ThreadPoolExecutor(cfg.episode_concurrency) as ex:
                results = list(ex.map(lambda j: (j[0], self.run_episode(sampler, j[1], cfg.temperature)), jobs))
        finally:
            sampler.close()

        groups: dict[int, list[dict[str, Any]]] = {}
        for gi, ep in results:
            groups.setdefault(gi, []).append(ep)
        all_eps = [ep for _, ep in results]

        datums: list[Any] = []
        kept_groups = 0
        for gi, eps in groups.items():
            rewards = [ep["score"]["reward"] for ep in eps]
            if len(set(rewards)) <= 1:
                continue
            kept_groups += 1
            for ep, adv in zip(eps, _group_relative_advantages(rewards)):
                for turn in ep["turns"]:
                    datums.append(self._datum(turn, adv))

        loss = None
        if datums:
            fb = self.training_client.forward_backward(datums, "importance_sampling").result()
            metrics = getattr(fb, "metrics", None) or {}
            loss_sum = metrics.get("loss:sum")   # the trainer reports a sum, not a per-token mean
            n_tok = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or 1.0
            loss = float(loss_sum) / max(float(n_tok), 1.0) if loss_sum is not None else None
            adam = tinker.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0)
            self.training_client.optim_step(adam).result()

        rec = {"step": step, "snapshot": snapshot, **_summarize("rollout", all_eps),
               "train/kept_groups": kept_groups, "train/groups": len(groups), "train/datums": len(datums),
               "train/loss": loss, "train/router_replay": self.router_replay_enabled,
               "perf/step_wall_time": time.time() - t0}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"step {step:02d} reward={rec['rollout/reward']:.3f} correct={rec['rollout/correct_rate']:.1%} "
            f"acted={rec['rollout/act_rate']:.1%} questions={rec['rollout/mean_questions']:.2f} "
            f"unneeded={rec['rollout/mean_unnecessary_questions']:.2f} groups={kept_groups}/{len(groups)} "
            f"datums={len(datums)} elapsed={rec['perf/step_wall_time']:.1f}s",
            flush=True,
        )
        return rec

    def _datum(self, turn: dict[str, Any], advantage: float) -> Any:
        prompt, tokens, logprobs = turn["prompt"], turn["tokens"], turn["logprobs"]
        response_start = prompt.length - 1
        model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
        if model_input.length > self.cfg.max_seq_len:
            raise ValueError(f"training datum exceeds max_seq_len ({model_input.length} > {self.cfg.max_seq_len})")
        if self.router_replay_enabled:
            validate_r3_routing_matrices(turn["routing"], prompt_len=prompt.length, model_input_len=model_input.length)
            aligned = build_r3_routing_matrices(turn["routing"], prompt_len=prompt.length,
                                                model_input_len=model_input.length, completion_only=True)
            model_input = model_input.model_copy(update=routing_model_input_kwargs(aligned))
        response_len = model_input.length - response_start
        return tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": [0] * response_start + tokens,
                "logprobs": [0.0] * response_start + logprobs,
                "advantages": [0.0] * response_start + [advantage] * response_len,
            },
        )

    # ----- run ---------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        snapshot0 = self.training_client.save_weights_for_sampler(f"{cfg.checkpoint_name}-base").result().path
        self.evaluate(snapshot0, 0)
        for step in range(cfg.steps):
            self.step(step)
            done = step + 1
            if cfg.eval_interval and done % cfg.eval_interval == 0 and done < cfg.steps:
                snap = self.training_client.save_weights_for_sampler(f"{cfg.checkpoint_name}-e{done:04d}").result().path
                self.evaluate(snap, done)
        final = self.training_client.save_weights_for_sampler(cfg.final_checkpoint_name).result().path
        print(f"final sampler checkpoint: {final}", flush=True)
        self.evaluate(final, cfg.steps)
        if cfg.output_model_id:
            self._promote(cfg.output_model_id)
        self.service.close()
        print(f"metrics: {self.metrics_path}", flush=True)

    def _promote(self, output_model_id: str) -> None:
        cp = FireworksClient(api_key=self.cfg.api_key, base_url=_control_plane_base_url(self.cfg.api_base_url))
        try:
            session = self.service.training_session_name or f"accounts/{cp.account_id}/trainingSessions/{self.service.training_session_id}"
            checkpoints = cp.list_training_session_checkpoints(session)
            match = _find_promotable(checkpoints, self.cfg.final_checkpoint_name, getattr(self.training_client, "run_id", None))
            if not match:
                print("final checkpoint not listed as promotable yet; promote manually", flush=True)
                return
            model = cp.promote_session_checkpoint(name=match["name"], output_model_id=output_model_id, base_model=self.cfg.base_model)
            print(f"promoted model: {model}", flush=True)
        finally:
            cp.close()


def _checkpoint_label(checkpoint: dict[str, Any]) -> str:
    return str(checkpoint.get("name", "")).rstrip("/").split("/")[-1]


def _find_promotable(checkpoints: list[dict[str, Any]], final_name: str, run_id: str | None) -> dict[str, Any] | None:
    """This run's final sampler row, matched by the server-owned id prefix so a re-run in the same
    session cannot promote an earlier run's checkpoint."""
    prefixes = [final_name] + ([f"{run_id}-{final_name}"] if run_id else [])

    def matches(checkpoint: dict[str, Any]) -> bool:
        label = _checkpoint_label(checkpoint)
        return any(label == prefix or label.startswith(prefix + "-") for prefix in prefixes)

    return next((row for row in checkpoints if matches(row) and bool(row.get("promotable"))), None)


def _transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        calls = [{"name": _tool_call_name(tc), "arguments": _tool_call_args(tc)} for tc in (m.get("tool_calls") or [])]
        out.append({"role": role, "content": content, "tool_calls": calls or None})
    return out


def _summarize(prefix: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(episodes))
    sc = [ep["score"] for ep in episodes]
    return {
        f"{prefix}/reward": sum(s["reward"] for s in sc) / n,
        f"{prefix}/correct_rate": sum(s["right_args"] for s in sc) / n,
        f"{prefix}/act_rate": sum(s["acted"] for s in sc) / n,
        f"{prefix}/right_tool_rate": sum(s["right_tool"] for s in sc) / n,
        f"{prefix}/mean_questions": sum(ep["questions"] for ep in episodes) / n,
        f"{prefix}/mean_unnecessary_questions": sum(s["unnecessary_questions"] for s in sc) / n,
        f"{prefix}/mean_turns": sum(len(ep["turns"]) for ep in episodes) / n,
        f"{prefix}/episodes": len(episodes),
        f"{prefix}/ended": {k: sum(ep["ended"] == k for ep in episodes) for k in sorted({ep["ended"] for ep in episodes})},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask Before Acting: multi-turn serverless RL (GRPO, importance sampling)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for f in Config.__dataclass_fields__.values():
        if f.name in ("api_key", "api_base_url"):
            continue
        default = getattr(Config, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            p.add_argument(flag, type=type(default), default=default)
    p.add_argument("--prepare-dataset", action="store_true", help="Generate the task file to --dataset and exit.")
    p.add_argument("--prepare-dataset-rows", type=int, default=600)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare_dataset:
        prepare_dataset(Path(args.dataset), args.prepare_dataset_rows, args.seed)
        return
    cfg = Config(**{k: v for k, v in vars(args).items() if k in Config.__dataclass_fields__})
    if len(cfg.final_checkpoint_name) > MAX_PROMOTABLE_CHECKPOINT_NAME_LEN:
        raise SystemExit(f"--final-checkpoint-name must be <= {MAX_PROMOTABLE_CHECKPOINT_NAME_LEN} chars to stay promotable")
    if not cfg.api_key:
        raise SystemExit("FIREWORKS_API_KEY is required")
    ServerlessAskBeforeActingRL(cfg).run()


if __name__ == "__main__":
    main()
