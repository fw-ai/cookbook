#!/usr/bin/env python3
"""NeMo Gym (any resources server + agent) -> Fireworks async RL.

Verified end-to-end against ``example_multi_step`` (real multi-turn tool use).
``--resources-server``/``--agent-name``/``--dataset-path`` point this at a
different NeMo Gym environment -- everything else (proxy, trajectory
tracking, Fireworks-side training) is environment-agnostic.

Architecture:
  - NeMo Gym owns all three servers -- resources/verifier, the agent harness
    that drives real multi-turn tool-calling rollouts, and the model server --
    plus the dataset and task orchestration. All genuinely NeMo Gym's, via a
    live `gym env start` process.
  - Fireworks' ``training.recipes.async_rl_loop.main()`` owns trainer/deployment
    lifecycle, rollout fan-out/admission, GRPO group assembly,
    forward/backward, the optimizer, sampler hotload, and checkpointing -- the
    same framework ``harbor_rl_opencode`` uses, not a hand-rolled loop.
  - ``rollout_fn`` (one call per individual sample, per the recipe's contract)
    makes a single direct HTTP POST to the harness's ``/run`` endpoint, setting
    NeMo Gym's own ``_ng_task_index``/``_ng_rollout_index`` fields and a unique
    ``_ng_rollout_id`` itself. The harness's model calls route through RecordingChatProxy
    (proxy.py), which samples from the recipe's Dedicated ``DeploymentSampler``
    and records exact token ids/logprobs per turn (``TrainingSessionTree``),
    keyed by that same correlation id (propagated through the `user` field via
    NeMo Gym's ``correlate_via_user_field`` -- NVIDIA-NeMo/Gym#3783).

Run:
  export FIREWORKS_API_KEY=fw_...
  python -m training.examples.rl.nemo_gym.train
  # against a different NeMo Gym env:
  python -m training.examples.rl.nemo_gym.train \\
      --resources-server my_env --agent-name my_env_simple_agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from training.renderer.model_info import get_recommended_renderer_name
from training.renderer.tokenizer import get_tokenizer

from training.examples.rl.nemo_gym.proxy import RecordingChatProxy
from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.recipes.async_rl_loop import Config, RolloutSetup, main
from training.utils import DeployConfig, TrainerConfig, WandBConfig
from training.utils.rl.rollout import RolloutRun

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
NEMO_GYM_DIR = Path(os.environ.get("NEMO_GYM_DIR", HERE.parents[4] / "nemo-gym")).resolve()

# Defaults match example_multi_step (the environment this was built and
# verified against); pass --resources-server/--agent-name/--dataset-path to
# point at a different NeMo Gym environment.
DEFAULT_RESOURCES_SERVER = "example_multi_step"
DEFAULT_AGENT_NAME = "example_multi_step_simple_agent"

# A provisionable training shape is needed for --base-model, so the default is
# pinned to a model/shape pair known to work (see --training-shape-id below).
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3p5-27b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.5-27B"

# NeMo Gym's head server (nemo_gym.global_config.DEFAULT_HEAD_SERVER_PORT) lists
# every server it started, with its URL, at GET /server_instances.
GYM_HEAD_SERVER_URL = "http://127.0.0.1:11000"


def _default_dataset_path(resources_server: str) -> Path:
    return NEMO_GYM_DIR / "resources_servers" / resources_server / "data" / "example.jsonl"


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ProxyThread:
    """Runs RecordingChatProxy on a dedicated, persistent event loop.

    A background aiohttp server must outlive whatever call started it --
    ``asyncio.run()`` inside the rollout factory would tear its loop down
    before any rollout ever reaches the server. Instead the proxy gets its own
    loop, on its own thread, alive for the whole process.
    """

    def __init__(self, proxy: RecordingChatProxy) -> None:
        self.proxy = proxy
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)

    def start(self, port: int) -> int:
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self.proxy.start(port), self._loop)
        return fut.result(timeout=30)

    def set_sampler(self, sampler: Any, sample_kwargs: dict[str, Any] | None = None) -> None:
        # The proxy awaits this sampler on this thread's loop, not the recipe's
        # -- see RecordingChatProxy.set_sampler for the constraint that implies.
        async def _set() -> None:
            self.proxy.set_sampler(sampler, sample_kwargs)

        asyncio.run_coroutine_threadsafe(_set(), self._loop).result(timeout=30)

    def stop(self) -> None:
        # Best-effort: in-flight requests still retrying against a torn-down
        # deployment can block AppRunner.cleanup() past any reasonable wait.
        # The paid trainer/deployment are already gone by this point (recipe's
        # own cleanup_on_exit) -- don't let a stuck HTTP retry hang the process.
        try:
            asyncio.run_coroutine_threadsafe(self.proxy.close(), self._loop).result(timeout=10)
        except Exception:
            logger.warning("proxy shutdown did not complete cleanly within 10s; forcing loop stop")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)


_GENERATED_HEADER = "# env.yaml -- generated by nemo_gym/train.py"


def write_nemo_gym_config(*, proxy_port: int) -> None:
    # observability_enabled=true is what makes NeMo Gym propagate its
    # rollout-correlation id into the outbound `user` field (the mechanism the
    # proxy keys sessions on); ModelCallCaptureConfig requires a capture dir
    # whenever it's on, even though we don't read NeMo Gym's own capture files
    # -- the proxy records exact token ids/logprobs itself.
    capture_dir = NEMO_GYM_DIR / "results" / "model_call_capture"
    env_path = NEMO_GYM_DIR / "env.yaml"
    if env_path.exists() and _GENERATED_HEADER not in env_path.read_text():
        raise SystemExit(
            f"{env_path} exists and was not generated by this example; refusing to overwrite it. "
            "Move it aside (or point NEMO_GYM_DIR at a clean checkout) and re-run."
        )
    env_path.write_text(
        f"{_GENERATED_HEADER}\n"
        f"policy_base_url: http://127.0.0.1:{proxy_port}/v1\n"
        'policy_api_key: "unused"\n'
        "policy_model_name: policy\n"
        "observability_enabled: true\n"
        f"model_call_capture_dir: {capture_dir}\n"
        # inference_provider.yaml has no ${...} interpolation hook for this
        # field, so it's overridden via a direct nested key -- OmegaConf.merge
        # (global_config.py) applies env.yaml over the whole config tree, not
        # just flat-key interpolation targets. Without this, NeMo Gym's own
        # Responses<->chat converter silently drops the model's
        # `reasoning_content` on every round trip (default is False), which
        # both discards real reasoning from NeMo Gym's own trajectory history
        # and breaks the proxy's turn continuation on turns that involved thinking.
        "policy_model:\n"
        "  responses_api_models:\n"
        "    inference_provider:\n"
        "      uses_reasoning_parser: true\n"
        # correlate_via_user_field is the NeMo Gym-side opt-in for exactly the
        # `user`-field propagation this proxy keys sessions on
        # (NVIDIA-NeMo/Gym#3783, merged 2026-10-08). On an older NeMo Gym this
        # key is rejected when `gym env start` validates env.yaml -- update it.
        "      correlate_via_user_field: true\n"
    )
    logger.info("wrote env.yaml pointing at the proxy (port %d)", proxy_port)


def start_gym_env(
    *, resources_server: str, agent_name: str, timeout_s: float, log_dir: Path
) -> tuple[subprocess.Popen, str]:
    """Launch `gym env start` and return (process, agent_base_url)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "gym_env_start.log"
    with log_path.open("w") as log_file:
        proc = _launch_gym_env(resources_server, log_file)
    logger.info("gym env start launched (pid=%d, log=%s), waiting for readiness...", proc.pid, log_path)
    try:
        return _wait_for_gym_env(proc, agent_name, log_path, timeout_s)
    except BaseException:
        # Don't leak the gym process (and its ray cluster) if startup fails.
        stop_gym_env(proc)
        raise


def _launch_gym_env(resources_server: str, log_file: Any) -> subprocess.Popen:
    return subprocess.Popen(
        [
            str(NEMO_GYM_DIR / ".venv" / "bin" / "gym"),
            "env",
            "start",
            "--resources-server",
            resources_server,
            "--model-type",
            # NOT "inference_provider/fireworks" -- that preset hardcodes
            # base_url to the real Fireworks API, ignoring env.yaml's
            # policy_base_url override. The generic "inference_provider"
            # preset reads ${policy_base_url} properly.
            "inference_provider",
            "-v",
        ],
        cwd=NEMO_GYM_DIR,
        env={**os.environ, "FIREWORKS_API_KEY": "unused-proxy-handles-auth"},
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def _http_up(url: str, timeout_s: float = 3.0) -> bool:
    """True if anything answers HTTP at ``url``, whatever the status.

    This is NeMo Gym's own readiness probe: a server is up once it responds at
    its base URL, even with a 404 (there may be no route at ``/``).
    """
    try:
        urllib.request.urlopen(url, timeout=timeout_s).close()
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _gym_server_urls(head_url: str) -> dict[str, str | None]:
    """``{server process name: base url}`` from the head server, or ``{}`` if it isn't up yet."""
    try:
        with urllib.request.urlopen(f"{head_url}/server_instances", timeout=3.0) as resp:
            instances = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    return {i.get("process_name"): i.get("url") for i in instances if i.get("process_name")}


def _wait_for_gym_env(
    proc: subprocess.Popen, agent_name: str, log_path: Path, timeout_s: float
) -> tuple[subprocess.Popen, str]:
    """Poll the head server until every NeMo Gym server answers HTTP; return the agent's URL."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"gym env start exited early; see {log_path}")
        urls = _gym_server_urls(GYM_HEAD_SERVER_URL)
        if urls and all(urls.values()) and all(_http_up(u) for u in urls.values()):
            agent_url = urls.get(agent_name)
            if agent_url is None:
                raise RuntimeError(f"agent {agent_name!r} is not among NeMo Gym's servers {sorted(urls)}")
            logger.info("nemo gym servers ready (%d); agent at %s", len(urls), agent_url)
            return proc, agent_url
        time.sleep(2)
    raise TimeoutError(f"gym env start did not become ready within {timeout_s:.0f}s; see {log_path}")


def stop_gym_env(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def build_rollout_fn_factory(
    *,
    proxy_thread: ProxyThread,
    agent_url: str,
    run_timeout_s: float,
):
    def make_rollout_fn(setup: "RolloutSetup"):
        sampler = build_deployment_sampler(setup)
        # RolloutSetup.sample_kwargs carries the recipe's on-policy sampling settings
        # (temperature, top_p=1.0, top_k=0, ...); the proxy must sample with the same ones.
        proxy_thread.set_sampler(sampler, setup.sample_kwargs)
        run_url = f"{agent_url.rstrip('/')}/run"

        async def rollout_fn(
            sample_prompt: dict,
            *,
            cursor_index: int,
            rollout_idx: int,
        ) -> RolloutRun | None:
            # An explicit, unique id per call: NeMo Gym uses it verbatim (instead of
            # deriving "{task}-{rollout}[-a{n}]" itself), and a retry with the same
            # indices can never append onto a stale session.
            rollout_id = f"{cursor_index}-{rollout_idx}-{uuid.uuid4().hex[:8]}"
            # Pass the whole dataset row through (matches rollout_collection.py's
            # own `json=row` pattern) rather than hand-picking fields -- the
            # row's own `id` (and any other agent-specific fields) are required
            # by validation but not knowable generically from here.
            body = {
                **sample_prompt,
                "_ng_task_index": cursor_index,
                "_ng_rollout_index": rollout_idx,
                "_ng_rollout_id": rollout_id,
            }
            t0 = time.monotonic()
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.post(
                        run_url,
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=run_timeout_s),
                    ) as resp:
                        resp.raise_for_status()
                        result = await resp.json()
            except BaseException:
                # /run failed or timed out: drop the partial session so it
                # doesn't accumulate for the lifetime of the process.
                proxy_thread.proxy.pop_session(rollout_id)
                raise
            session = proxy_thread.proxy.pop_session(rollout_id)
            if session is None:
                logger.warning("no proxy session recorded for rollout_id=%s", rollout_id)
                return None
            samples = session.to_samples(reward=float(result["reward"]))
            logger.info(
                "rollout %s: %d turns -> %d training segment(s), %d trained tokens, reward=%s (%.1fs)",
                rollout_id,
                session.turn_count,
                len(samples),
                sum(sum(s.loss_mask) for s in samples),
                result.get("reward"),
                time.monotonic() - t0,
            )
            return RolloutRun(segments=samples) if samples else None

        return rollout_fn

    return make_rollout_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeMo Gym multi-step async RL (Dedicated deployment)")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    p.add_argument("--renderer-name", default="")
    p.add_argument(
        "--resources-server",
        default=DEFAULT_RESOURCES_SERVER,
        help="NeMo Gym resources server to run (passed to `gym env start`).",
    )
    p.add_argument(
        "--agent-name",
        default=DEFAULT_AGENT_NAME,
        help="NeMo Gym agent process name (used to find its URL in gym env start's log).",
    )
    p.add_argument(
        "--dataset-path",
        default=None,
        help="Defaults to resources_servers/<resources-server>/data/example.jsonl.",
    )
    p.add_argument("--output-model-id", default=None)
    p.add_argument("--max-rows", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--completions-per-prompt", type=int, default=4)
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prompt-groups-per-step", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2.5e-5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument(
        "--training-shape-id",
        # Auto-selection (None) failed live: "Cannot create a managed
        # deployment without a deployment shape" -- for this model/account it
        # doesn't resolve a linked deployment shape on its own. Pin the shape
        # the live catalog lists for LoRA RL on qwen3p5-27b.
        default="accounts/fireworks/trainingShapes/qwen3p5-27b-64k-lora",
    )
    p.add_argument("--replica-count", type=int, default=None)
    p.add_argument("--max-concurrency-rollout-sample", type=int, default=None)
    p.add_argument(
        "--exclude-user-simulator",
        action="store_true",
        help="Keep toolsandbox-style simulated-user model calls out of the training chain "
        "(heuristic: calls with no tools or only `end_conversation`). Off by default.",
    )
    p.add_argument(
        "--gym-start-timeout-s",
        type=float,
        default=900.0,
        help="Readiness timeout for `gym env start`. A first start on a fresh NeMo Gym checkout "
        "builds one venv per server and can take many minutes.",
    )
    p.add_argument("--proxy-port", type=int, default=18234)
    p.add_argument(
        "--run-timeout-s",
        type=float,
        default=300.0,
        # DeploymentSampler retries internally (its own backoff budget, e.g.
        # up to 10x "not ready" + 7x transient-5xx retries after a fresh
        # hotload) -- a short outer timeout here cuts that budget off from
        # outside and looks like a rollout failure even though the sampler
        # would have succeeded given time (e.g. the post-hotload warm-up window).
        help="Per-rollout HTTP timeout to the NeMo Gym agent's /run endpoint.",
    )
    p.add_argument(
        "--log-path",
        default="./nemo_gym_logs",
        help="Directory for the recipe's logs and for gym_env_start.log (NeMo Gym's startup output).",
    )
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "nemo-gym-multistep"))
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def run() -> None:
    if "FIREWORKS_API_KEY" not in os.environ:
        raise SystemExit("FIREWORKS_API_KEY is required")
    args = parse_args()
    dataset_path = Path(args.dataset_path) if args.dataset_path else _default_dataset_path(args.resources_server)

    rows = _load_rows(dataset_path)[: args.max_rows]
    if not rows:
        raise SystemExit(f"dataset is empty: {dataset_path}")
    logger.info("loaded %d NeMo Gym rows from %s", len(rows), dataset_path)

    tokenizer = get_tokenizer(args.tokenizer_model)
    renderer_name = args.renderer_name or get_recommended_renderer_name(args.tokenizer_model)

    proxy = RecordingChatProxy(
        tokenizer=tokenizer,
        renderer_name=renderer_name,
        max_sample_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        exclude_user_simulator=args.exclude_user_simulator,
    )
    proxy_thread = ProxyThread(proxy)
    bound_port = proxy_thread.start(args.proxy_port)
    logger.info("proxy listening on 127.0.0.1:%d", bound_port)

    write_nemo_gym_config(proxy_port=bound_port)
    gym_proc, agent_url = start_gym_env(
        resources_server=args.resources_server,
        agent_name=args.agent_name,
        timeout_s=args.gym_start_timeout_s,
        log_dir=Path(args.log_path),
    )

    try:
        make_rollout_fn = build_rollout_fn_factory(
            proxy_thread=proxy_thread,
            agent_url=agent_url,
            run_timeout_s=args.run_timeout_s,
        )
        run_name = args.wandb_run_name or f"nemo-gym-multistep-{int(time.time()) % 100000}"
        cfg = Config(
            log_path=args.log_path,
            base_model=args.base_model,
            learning_rate=args.learning_rate,
            completions_per_prompt=args.completions_per_prompt,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            epochs=args.epochs,
            max_rows=args.max_rows,
            lora_rank=args.lora_rank,
            prompt_groups_per_step=args.prompt_groups_per_step,
            max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
            output_model_id=args.output_model_id,
            trainer=TrainerConfig(training_shape_id=args.training_shape_id),
            deployment=DeployConfig(
                tokenizer_model=args.tokenizer_model,
                replica_count=args.replica_count,
            ),
            wandb=WandBConfig(
                entity=args.wandb_entity,
                project=args.wandb_project,
                run_name=run_name,
            ),
        )
        main(cfg, rollout_fn_factory=make_rollout_fn, rows=rows, rollout_extras={})
    finally:
        stop_gym_env(gym_proc)
        proxy_thread.stop()


if __name__ == "__main__":
    run()
