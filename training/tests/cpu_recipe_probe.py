"""Fixed cookbook training step through the real SDK and loopback HTTP.

No GPU, inference, provisioning or credentials. The trainer is a deterministic
HTTP fixture; cookbook preparation, SDK scheduling/encoding/futures, decoding,
loss metrics and optimizer RPCs are real. Timed runs exclude profiler overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import socket
import statistics
import threading
import time
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fireworks.training import sdk
from fireworks.training.sdk import client as sdk_client
from training.recipes import rl_loop
from training.utils.client import ReconnectableClient
from training.utils.phase_tracing import (
    configure_phase_tracing,
    flush_phase_trace,
    phase_span,
)
from training.utils.rl.common import inference_drift_metrics
from training.utils.rl.losses import PolicyBatch, build_policy_datums
from training.utils.rl.rollout import RolloutRun, RolloutSample

runtime = sdk_client.ServiceClient.__module__.split(".lib.public_interfaces")[0]
base = importlib.import_module(runtime + "._base_client")
holder_module = importlib.import_module(runtime + ".lib.internal_client_holder")
conv = importlib.import_module(runtime + ".lib._pydantic_conv")


class TrainerFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pending: dict[str, bytes] = {}
    calls: list[dict] = []
    lock = threading.Lock()

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *args):
        pass

    def do_POST(self):
        operation = self.path.rsplit("/", 1)[-1]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append({"operation": operation, "body": body})
        if operation == "create_session":
            result = {"session_id": "session-fixed"}
        elif operation == "retrieve_future":
            payload = self.pending[body["request_id"]]
            self.respond(payload)
            return
        elif operation in ("telemetry", "session_heartbeat"):
            result = {}
        else:
            with self.lock:
                rid = "result-" + str(len(self.pending))
                self.pending[rid] = b""
            if operation == "create_model":
                value = {"model_id": "model-fixed", "comms": "v2"}
            elif operation in ("forward", "forward_backward"):
                inputs = body.get("forward_input", body.get("forward_backward_input"))["data"]
                value = {
                    "loss_fn_output_type": "ppo",
                    "metrics": {"loss:sum": 1.0},
                    "loss_fn_outputs": [{"logprobs": datum["loss_fn_inputs"]["logprobs"]} for datum in inputs],
                }
            elif operation == "optim_step":
                value = {"metrics": {"step_id:last": 1.0}}
            elif operation in ("save_weights", "save_weights_for_sampler"):
                value = {"path": body["path"]}
            else:
                raise AssertionError(operation)
            self.pending[rid] = json.dumps(value, separators=(",", ":")).encode()
            result = {"request_id": rid}
        self.respond(json.dumps(result, separators=(",", ":")).encode())

    def respond(self, payload):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def make_batch(rows: int, tokens: int) -> PolicyBatch:
    rng = np.random.default_rng(42)
    datums, anchors = [], []
    for _ in range(rows):
        ids = rng.integers(0, 32000, tokens + 1, dtype=np.int64)
        logprobs = rng.normal(-2.0, 0.1, tokens).astype(np.float32).tolist()
        datums.append(
            sdk.Datum(
                model_input=sdk.ModelInput.from_ints(ids[:-1].tolist()),
                loss_fn_inputs={"target_tokens": ids[1:], "weights": np.ones(tokens, dtype=np.float32)},
            )
        )
        anchors.append(logprobs)
    return PolicyBatch(
        data=datums,
        advantages=[(-1.0 if i % 2 else 1.0) for i in range(rows)],
        prompt_lens=[tokens // 4] * rows,
        rollout_logprobs=anchors,
    )


def train_step(policy, batch):
    durations = {}

    @contextmanager
    def stage(name):
        started = time.perf_counter_ns()
        with phase_span(name, category="cookbook"):
            yield
        durations[name] = (time.perf_counter_ns() - started) / 1e6

    with stage("cookbook_build_policy_datums"):
        datums = build_policy_datums(batch)
    with stage("forward_backward_rpc_and_result"):
        result = policy.forward_backward(datums, "ppo", loss_fn_config={"clip_eps": 0.2})
    with stage("cookbook_drift_metrics"):
        metrics = inference_drift_metrics(batch, result)
        assert all(np.isfinite(value) for value in metrics.values()), metrics
    with stage("optimizer_rpc_and_result"):
        policy.optim_step(sdk.AdamParams(learning_rate=0.0))
    assert len(result.loss_fn_outputs) == len(datums), (len(result.loss_fn_outputs), len(datums))
    return durations


def recipe_step(service, client, batch, output):
    """Run the actual recipe; only provisioning, inference and log I/O are fixtures."""
    events = []
    sampler = SimpleNamespace(
        base_url="http://fixture.invalid", model="test/model", drain_metrics=lambda: [], close=lambda: None
    )

    class InfrastructureFixture:
        trainer_job_id = "test"
        deployment_id = "fixture"
        max_context_length = len(batch.rollout_logprobs[0]) + 1
        accelerator_type = "fixture"
        accelerator_count = 0

        def create_training_client(self, *a, **kw):
            return client  # Reattach the same initialized model for timed runs.

        def create_deployment_sampler(self, **kwargs):
            return sampler

        def close(self):
            pass  # The probe owns the real service across timed runs.

    class CheckpointFixture:
        def __init__(self, policy, *a, **kw):
            self.policy = policy

        def resume(self, **kw):
            return None

        def save(self, name, **kw):
            self.policy.save_state(name)

    def publisher(policy, *a):
        return lambda name, **kw: policy.save_weights_for_sampler(name, **kw)

    def rollout_factory(setup):
        async def rollout(row, *, rollout_idx, **kw):
            i = row["group"] * 2 + rollout_idx
            datum = batch.data[i]
            tokens = datum.model_input.to_ints() + [int(datum.loss_fn_inputs["target_tokens"].data[-1])]
            sample = RolloutSample(
                tokens=tokens,
                logprobs=[0.0] + batch.rollout_logprobs[i],
                loss_mask=[0] * batch.prompt_lens[i] + [1] * (len(tokens) - batch.prompt_lens[i]),
                reward=batch.advantages[i],
            )
            return RolloutRun(segments=[sample])

        return rollout

    with ExitStack() as stack:
        stack.enter_context(patch.dict("os.environ", {"FIREWORKS_API_KEY": "fw-local-fixture"}))
        for name, replacement in {
            "build_service_client": lambda **_kw: InfrastructureFixture(),
            "load_deployment_tokenizer": lambda *_a: object(),
            "resolve_router_replay_enabled": lambda **_kw: False,
            "validate_config": lambda *_a, **_kw: None,
            "setup_wandb": lambda *_a, **_kw: None,
            "wandb_finish": lambda **_kw: None,
            "log_metrics": lambda metrics, **_kw: events.append(dict(metrics)),
            "log_metrics_json": lambda *_a, **_kw: None,
            "make_weight_sync": publisher,
            "TrainingCheckpoints": CheckpointFixture,
        }.items():
            stack.enter_context(patch.object(rl_loop, name, replacement))
        start = time.perf_counter_ns()
        result = rl_loop.main(
            rl_loop.Config(
                base_model="test/model",
                lora_rank=8,
                learning_rate=0.0,
                log_path=str(output.parent),
                max_seq_len=InfrastructureFixture.max_context_length,
                completions_per_prompt=2,
                prompt_groups_per_step=len(batch.data) // 2,
                epochs=1,
                shuffle=False,
                save_final_checkpoint=True,
                deployment=rl_loop.DeployConfig(tokenizer_model="fixture"),
            ),
            rows=[{"group": i} for i in range(len(batch.data) // 2)],
            rollout_fn_factory=rollout_factory,
        )
        elapsed = (time.perf_counter_ns() - start) / 1e6
        assert result["steps"] == 1, result
    return {"recipe_e2e": elapsed}


def main():
    import yappi

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--recipe", action="store_true")
    args = parser.parse_args()
    if args.rows < 2 or args.rows % 2 or args.tokens < 4:
        parser.error("rows must be positive and even; tokens must be >= 4")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch = make_batch(args.rows, args.tokens)
    server = ThreadingHTTPServer(("127.0.0.1", 0), TrainerFixture)
    server_thread = threading.Thread(target=server.serve_forever, name="fixture-server", daemon=True)
    server_thread.start()
    transport = base.AsyncHttpxClientWrapper()
    with patch.object(holder_module, "init_telemetry", return_value=None):
        service = sdk_client.FiretitanServiceClient(
            api_key="fw-local-fixture",
            base_url=f"http://127.0.0.1:{server.server_port}/training/v1/rlorTrainerJobs/test/trainer",
            http_client=transport,
            max_retries=0,
            _client_config={"use_pyqwest_transport": False, "parallel_fwdbwd_chunks": True},
        )
    service._resolved_account_id = lambda: "test"
    samples = []
    try:
        client = service.create_training_client(base_model="test/model", lora_rank=8, seed=42)
        policy = ReconnectableClient.from_training_client(client, base_model="test/model", job_id="test")
        step = (
            (lambda: recipe_step(service, client, batch, output))
            if args.recipe
            else (lambda: train_step(policy, batch))
        )
        for i in range(12):
            timings = step()
            if i >= 3:
                samples.append(timings)
        configure_phase_tracing(output.with_suffix(".trace.json"))

        # Instrument just the benchmark's owned request boundary, and restore it.
        def traced(fn, name):
            def wrapped(*a, **kw):
                with phase_span(name, category="sdk"):
                    return fn(*a, **kw)

            return wrapped

        yappi.set_clock_type("cpu")
        yappi.start(builtins=True)
        with (
            patch.object(conv, "_to_pydantic_datum", traced(conv._to_pydantic_datum, "sdk_datum_to_wire_schema")),
            patch.object(
                base.AsyncHttpxClientWrapper,
                "build_request",
                traced(base.AsyncHttpxClientWrapper.build_request, "sdk_build_http_request"),
            ),
            patch.object(
                rl_loop.objective, "build_datums", traced(rl_loop.objective.build_datums, "cookbook_build_datums")
            ),
            patch.object(rl_loop.objective, "metrics", traced(rl_loop.objective.metrics, "cookbook_grpo_metrics")),
            patch.object(
                rl_loop, "inference_drift_metrics", traced(rl_loop.inference_drift_metrics, "cookbook_drift_metrics")
            ),
        ):
            step()
        yappi.stop()
        yappi.get_func_stats().save(str(output.with_suffix(".pstats")), type="pstat")
        flush_phase_trace()
        report = {
            "sdk_source": sdk_client.__file__,
            "workload": [args.rows, args.tokens],
            "transport": "real loopback HTTP/1.1; deterministic trainer fixture",
            "warmups": 3,
            "runs": 9,
            "samples_ms": samples,
            "median_ms": {k: statistics.median(v[k] for v in samples) for k in samples[0]},
            "threads": [{"name": s.name, "cpu_s": s.ttot} for s in yappi.get_thread_stats()],
        }
        bodies = [
            call["body"]["forward_backward_input"]
            for call in sorted(
                [c for c in TrainerFixture.calls if c["operation"] == "forward_backward"],
                key=lambda c: c["body"]["seq_id"],
            )
        ]
        report["training_inputs_sha256"] = hashlib.sha256(
            json.dumps(bodies, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["median_ms"]))
    finally:
        service.close()
        service.holder.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
