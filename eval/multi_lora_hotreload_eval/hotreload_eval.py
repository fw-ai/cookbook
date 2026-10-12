"""Multi-LoRA evaluation reusing one hot-swap deployment.

    1. Create ONE deployment with `enableHotReloadLatestAddon=true` on a base model.
       `deploymentId` is passed as a query param (?deploymentId=...), NOT in the
       body. A `deploymentShape` is required — the shape owns the accelerator
       type/count and serving config, so no separate accelerator fields are sent.
    2. For each LoRA in the list:
       a. POST /v1/accounts/{acct}/deployedModels?replaceMergedAddon=true
          body: {model: accounts/<acct>/models/<lora>, deployment: accounts/<acct>/deployments/<dep>}
       b. Poll GetDeployment until READY (confirms the deployment is healthy).
       c. Poll GetDeployedModel until the addon state == DEPLOYED. The deployment
          can flip READY *before* the new merge is actually serving, so the addon
          state is the real readiness signal.
       d. Warmup-probe the deployment route with a 1-token request until it
          stops 404ing (the route can briefly 404 right after the addon is
          DEPLOYED). Inference MUST address the deployment route, not the
          adapter's model id.
       e. Run an evaluation against the deployment route.
       f. Save eval results.
    3. Clean up the deployment. If a plain DELETE fails, retry with
       ?ignoreChecks=true to avoid leaking a live deployment.


Usage:
    export FIREWORKS_ACCOUNT_NAME=...
    export FIREWORKS_API_KEY=...
    python hotreload_eval.py \
        --base-model accounts/fireworks/models/qwen3-8b \
        --deployment-shape accounts/fireworks/deploymentShapes/qwen3-8b-minimal \
        --deployment-id multi-lora-eval-$(date +%s) \
        --lora-models accounts/fireworks/models/lora-a,accounts/fireworks/models/lora-b \
        --eval-dataset ./eval.jsonl \
        --results-dir ./results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAI

log = logging.getLogger("hotreload_eval")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class EvalConfig:
    account_id: str
    base_model: str  # accounts/<acct>/models/<base>
    deployment_id: str  # short id, will be expanded to accounts/<acct>/deployments/<id>
    lora_models: list[str]  # each: accounts/<acct>/models/<lora>
    eval_dataset_path: str  # JSONL with {"prompt": "...", ...} per line
    results_dir: str
    api_base: str = "https://api.fireworks.ai"
    poll_interval_s: int = 5
    poll_timeout_s: int = 600  # per LoRA load
    create_deployment: bool = True  # if False, reuse an existing deployment_id
    cleanup_deployment: bool = True
    # Inference params
    max_tokens: int = 256
    temperature: float = 0.0
    # Deployment shape — REQUIRED. The shape owns the hardware (accelerator
    # type/count) and serving config, so raw REST CreateDeployment does not
    # need (and must not send) acceleratorType/acceleratorCount.
    min_replica_count: int | None = 1
    max_replica_count: int | None = 1
    deployment_shape: str = ""  # accounts/<acct>/deploymentShapes/<id>
    # Warmup probing: after an addon swap, how long to keep probing the
    # deployment route until it stops 404ing, and how often.
    warmup_timeout_s: int = 180
    warmup_interval_s: int = 3


# -----------------------------------------------------------------------------
# Fireworks REST client (thin)
# -----------------------------------------------------------------------------


class FireworksClient:
    """Thin REST wrapper around the deployment REST endpoints used here.

    Raw REST is used rather than the public `fireworks` SDK because the SDK
    does not expose the `replaceMergedAddon` query parameter, which is the
    whole point of the hot-reload path.
    """

    def __init__(self, account_id: str, api_key: str, api_base: str):
        self.account_id = account_id
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def deployment_name(self, deployment_id: str) -> str:
        return f"accounts/{self.account_id}/deployments/{deployment_id}"

    def create_deployment(
        self, body: dict[str, Any], *, deployment_id: str | None = None
    ) -> dict[str, Any]:
        # NOTE: `deploymentId` is a *query* parameter on CreateDeployment, not
        # a body field. Putting it in the body is rejected by the API.
        params: dict[str, str] = {}
        if deployment_id:
            params["deploymentId"] = deployment_id
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployments"
        r = self._client.post(url, params=params, json=body)
        r.raise_for_status()
        return r.json()

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployments/{deployment_id}"
        r = self._client.get(url)
        r.raise_for_status()
        return r.json()

    def delete_deployment(self, deployment_id: str, *, ignore_checks: bool = False) -> None:
        # A plain DELETE can be rejected by the API (e.g. while the deployment
        # is still in use). Passing ?ignoreChecks=true forces it, so we retry
        # with that on failure to avoid leaking live deployments.
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployments/{deployment_id}"
        r = self._client.delete(url, params={"ignoreChecks": "true"} if ignore_checks else None)
        if r.status_code == 404:
            # Already gone — treat as success.
            return
        if not r.is_success and not ignore_checks:
            log.info(
                "plain delete failed (%s); retrying with ?ignoreChecks=true", r.status_code
            )
            r2 = self._client.delete(url, params={"ignoreChecks": "true"})
            if r2.status_code == 404:
                return
            r2.raise_for_status()
            return
        r.raise_for_status()

    def load_lora_hot_reload(self, lora_model_name: str, deployment_name: str) -> dict[str, Any]:
        """POST /v1/accounts/{acct}/deployedModels?replaceMergedAddon=true

        Requests that the deployment merge the given LoRA in-place as its
        active addon, replacing any previously merged addon. Returns the
        created deployedModel (its `name` is the addon resource name used to
        poll state).
        """
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployedModels"
        r = self._client.post(
            url,
            params={"replaceMergedAddon": "true"},
            json={
                "model": lora_model_name,
                "deployment": deployment_name,
            },
        )
        # AlreadyExists (409): the LoRA is already the merged addon. Resolve
        # its deployedModel name via list so its state can still be polled.
        if r.status_code == 409:
            addon = self._find_deployed_model(lora_model_name, deployment_name)
            if addon is None:
                raise RuntimeError(
                    f"load_lora got 409 but could not find existing deployedModel for "
                    f"{lora_model_name} on {deployment_name}"
                )
            return {"alreadyExists": True, **addon}
        r.raise_for_status()
        return r.json()

    def get_deployed_model(self, deployed_model_id: str) -> dict[str, Any]:
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployedModels/{deployed_model_id}"
        r = self._client.get(url)
        r.raise_for_status()
        return r.json()

    def _find_deployed_model(
        self, lora_model_name: str, deployment_name: str
    ) -> dict[str, Any] | None:
        """List deployedModels and return the one matching (model, deployment)."""
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployedModels"
        token: str | None = None
        while True:
            params: dict[str, Any] = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            r = self._client.get(url, params=params)
            r.raise_for_status()
            page = r.json()
            for dm in page.get("deployedModels", []):
                if dm.get("model") == lora_model_name and dm.get("deployment") == deployment_name:
                    return dm
            token = page.get("nextPageToken") or None
            if not token:
                return None

    def close(self) -> None:
        self._client.close()


# -----------------------------------------------------------------------------
# Hot-reload eval workflow
# -----------------------------------------------------------------------------


@dataclass
class LoRAEvalResult:
    lora_model: str
    deployment_state: str
    loaded_in_s: float
    n_samples: int
    outputs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class MultiLoraHotReloadEvaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        api_key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FW_API_KEY")
        if not api_key:
            sys.exit("Set FIREWORKS_API_KEY (or FW_API_KEY) in the environment.")
        self.cp = FireworksClient(cfg.account_id, api_key, cfg.api_base)
        # OpenAI-compatible client pointed at Fireworks. For hot-reload
        # deployments, address the deployment route (accounts/<acct>/
        # deployments/<id>) as the `model` field — it resolves to the currently
        # merged addon. Addressing the adapter's model id is not supported.
        self.oai = OpenAI(
            base_url=f"{cfg.api_base}/inference/v1",
            api_key=api_key,
        )

    # -- deployment lifecycle --------------------------------------------------

    def create_hot_reload_deployment(self) -> dict[str, Any]:
        if not self.cfg.deployment_shape:
            raise RuntimeError(
                "deployment_shape is required (the shape owns accelerator type/count "
                "and serving config, so raw REST does not need separate accelerator fields)."
            )
        body: dict[str, Any] = {
            "baseModel": self.cfg.base_model,
            "deploymentShape": self.cfg.deployment_shape,
            "enableHotReloadLatestAddon": True,
            "enableHotLoad": True,
            "hotLoadBucketType": "FW_HOSTED",
        }
        if self.cfg.min_replica_count is not None:
            body["minReplicaCount"] = self.cfg.min_replica_count
        if self.cfg.max_replica_count is not None:
            body["maxReplicaCount"] = self.cfg.max_replica_count

        log.info("[create] POST /deployments?deploymentId=%s body=%s", self.cfg.deployment_id, json.dumps(body))
        dep = self.cp.create_deployment(body, deployment_id=self.cfg.deployment_id)
        log.info("[create] deployment=%s state=%s", dep.get("name"), dep.get("state"))
        return dep

        
    def create_hot_reload_deployment(base_model, deployment_shape) -> dict[str, Any]:
        body: dict[str, Any] = {
            "baseModel": base_model,
            "deploymentShape": deployment_shape,
            "enableHotReloadLatestAddon": True,
            "enableHotLoad": True,
            "hotLoadBucketType": "FW_HOSTED",
        }

        log.info("[create] POST /deployments?deploymentId=%s body=%s", self.cfg.deployment_id, json.dumps(body))
        dep = self.cp.create_deployment(body, deployment_id=self.cfg.deployment_id)
        log.info("[create] deployment=%s state=%s", dep.get("name"), dep.get("state"))
        return dep

    def wait_for_deployment_ready(self, deployment_id: str) -> dict[str, Any]:
        """Poll GetDeployment until READY.

        The deployment's READY state alone is NOT sufficient to start eval
        after an addon swap — the deployment can flip READY before the route
        actually serves the new merge. The real readiness signal is the
        addon's DEPLOYED state plus a warmup probe of the route, both of
        which are performed by ``load_lora`` after this returns. This poll
        only guards the initial bring-up and confirms the deployment is not
        in a failed state.
        """
        deadline = time.time() + self.cfg.poll_timeout_s
        last_state: str | None = None
        while time.time() < deadline:
            dep = self.cp.get_deployment(deployment_id)
            state = dep.get("state")
            if state != last_state:
                log.info("[poll] %s: state=%s", deployment_id, state)
                last_state = state
            if state == "READY":
                return dep
            if state in {"FAILED", "DELETED", "DELETING", "STATE_UNSPECIFIED"}:
                raise RuntimeError(f"deployment entered bad state: {state}")
            time.sleep(self.cfg.poll_interval_s)
        raise TimeoutError(
            f"deployment {deployment_id} never reached READY within {self.cfg.poll_timeout_s}s"
        )

    @staticmethod
    def _deployed_model_id_from_name(name: str) -> str:
        """accounts/<acct>/deployedModels/<id> -> <id>"""
        return name.rsplit("/", 1)[-1]

    def wait_for_addon_deployed(self, addon_name: str) -> None:
        """Poll GetDeployedModel until state == DEPLOYED.

        The deployment can flip to READY *before* the newly swapped addon is
        actually serving on the deployment route, so polling the deployment
        state alone is not enough. The addon's own state transitions
        DEPLOYING/UPDATING -> DEPLOYED once the merge is live, which is the
        correct readiness signal.
        """
        addon_id = self._deployed_model_id_from_name(addon_name)
        deadline = time.time() + self.cfg.poll_timeout_s
        last_state: str | None = None
        while time.time() < deadline:
            dm = self.cp.get_deployed_model(addon_id)
            state = dm.get("state")
            if state != last_state:
                log.info("[addon] %s: state=%s", addon_id, state)
                last_state = state
            if state == "DEPLOYED":
                return
            if state == "STATE_UNSPECIFIED":
                raise RuntimeError(f"addon {addon_id} entered bad state: {state}")
            time.sleep(self.cfg.poll_interval_s)
        raise TimeoutError(
            f"addon {addon_id} never reached DEPLOYED within {self.cfg.poll_timeout_s}s"
        )

    def warmup_probe_route(self) -> None:
        """Probe the deployment route with a 1-token request until it succeeds.

        Even after the addon is DEPLOYED, the route can keep 404ing for a few
        seconds before it starts serving the new merge. A minimal completion
        is sent to warm the route; once one succeeds, eval can proceed against
        the actually-serving merge.

        For hot-reload deployments the request MUST address the deployment
        route (accounts/<acct>/deployments/<id>), not the adapter's model id.
        """
        deployment_route = self.cp.deployment_name(self.cfg.deployment_id)
        deadline = time.time() + self.cfg.warmup_timeout_s
        attempt = 0
        last_err: str | None = None
        while time.time() < deadline:
            attempt += 1
            try:
                self.oai.completions.create(
                    model=deployment_route,
                    prompt="hi",
                    max_tokens=1,
                    temperature=0.0,
                )
                log.info("[warmup] route serving after %d probe(s)", attempt)
                return
            except Exception as e:
                last_err = str(e)
                if time.time() >= deadline:
                    break
                time.sleep(self.cfg.warmup_interval_s)
        raise TimeoutError(
            f"deployment route {deployment_route} never stopped 404ing within "
            f"{self.cfg.warmup_timeout_s}s (last error: {last_err})"
        )

    def load_lora(self, lora_model_name: str) -> float:
        deployment_name = self.cp.deployment_name(self.cfg.deployment_id)
        log.info("[load] POST /deployedModels?replaceMergedAddon=true model=%s -> %s", lora_model_name, deployment_name)
        t0 = time.time()
        resp = self.cp.load_lora_hot_reload(lora_model_name, deployment_name)
        addon_name = resp.get("name")
        if not addon_name:
            raise RuntimeError(
                f"load_lora response missing 'name' (deployedModel resource name): {resp}"
            )
        log.info("[load] addon=%s alreadyExists=%s", addon_name, resp.get("alreadyExists", False))

        # 1. Wait for the deployment to be READY. This alone is not enough to
        #    start eval (the route can 404 right after READY), but it confirms
        #    the deployment is healthy and not in a failed state. The real
        #    readiness gates are the addon DEPLOYED state + the route warmup
        #    probe below, which also cover the 409 already-merged case (where
        #    the deployment never goes UPDATING).
        self.wait_for_deployment_ready(self.cfg.deployment_id)
        # 2. Wait for the addon itself to be DEPLOYED. The deployment can flip
        #    READY before the route serves the new merge, so this is the real
        #    readiness signal.
        self.wait_for_addon_deployed(addon_name)
        # 3. Warmup-probe the deployment route until it stops 404ing, closing
        #    the remaining gap before eval runs.
        self.warmup_probe_route()

        elapsed = time.time() - t0
        log.info("[load] ready in %.1fs", elapsed)
        return elapsed

    # -- eval ------------------------------------------------------------------

    def load_eval_dataset(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with open(self.cfg.eval_dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        log.info("[eval] loaded %d samples from %s", len(samples), self.cfg.eval_dataset_path)
        return samples

    def run_eval(self, lora_model_name: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # For hot-reload deployments, the deployment route resolves to the
        # currently merged addon. Use the full deployment resource name.
        deployment_route = self.cp.deployment_name(self.cfg.deployment_id)
        outputs: list[dict[str, Any]] = []
        for i, s in enumerate(samples):
            prompt = s.get("prompt") or s.get("messages")
            if prompt is None:
                raise ValueError(f"sample {i} missing 'prompt' or 'messages'")
            kwargs: dict[str, Any] = {
                "model": deployment_route,
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
            }
            if isinstance(prompt, str):
                kwargs["prompt"] = prompt
            else:
                kwargs["messages"] = prompt
            try:
                if "prompt" in kwargs:
                    r = self.oai.completions.create(**kwargs)
                    text = r.choices[0].text
                else:
                    r = self.oai.chat.completions.create(**kwargs)
                    text = r.choices[0].message.content
                outputs.append({"sample_id": i, "output": text, "raw": s})
            except Exception as e:
                outputs.append({"sample_id": i, "error": str(e), "raw": s})
        return outputs

    # -- cleanup ---------------------------------------------------------------

    def cleanup(self) -> None:
        if not self.cfg.cleanup_deployment:
            return
        log.info("[cleanup] deleting deployment %s", self.cfg.deployment_id)
        try:
            self.cp.delete_deployment(self.cfg.deployment_id)
        except Exception as e:
            log.error("[cleanup] failed: %s", e)

    # -- top-level -------------------------------------------------------------

    def run(self) -> list[LoRAEvalResult]:
        os.makedirs(self.cfg.results_dir, exist_ok=True)
        samples = self.load_eval_dataset()

        results: list[LoRAEvalResult] = []
        try:
            # Create + initial readiness wait live INSIDE the try/finally so
            # that a timeout or failure during initial bring-up still triggers
            # cleanup and avoids leaking the deployment.
            if self.cfg.create_deployment:
                self.create_hot_reload_deployment()
                self.wait_for_deployment_ready(self.cfg.deployment_id)

            for lora in self.cfg.lora_models:
                res = LoRAEvalResult(
                    lora_model=lora,
                    deployment_state="",
                    loaded_in_s=0.0,
                    n_samples=len(samples),
                )
                try:
                    res.loaded_in_s = self.load_lora(lora)
                    res.deployment_state = "READY"
                    # 
                    res.outputs = self.run_eval(lora, samples)
                except Exception as e:
                    res.error = str(e)
                    log.error("[run] LoRA %s failed: %s", lora, e)
                results.append(res)
                self._write_lora_results(res)
        finally:
            self.cleanup()
        return results

    def _write_lora_results(self, res: LoRAEvalResult) -> None:
        lora_id = res.lora_model.split("/")[-1]
        out_path = os.path.join(self.cfg.results_dir, f"{lora_id}.jsonl")
        with open(out_path, "w") as f:
            for o in res.outputs:
                f.write(json.dumps(o) + "\n")
        log.info("[results] wrote %s", out_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--account-id", default=os.environ.get("FIREWORKS_ACCOUNT_NAME") or os.environ.get("FW_ACCOUNT_ID"))
    p.add_argument("--base-model", required=True, help="accounts/<acct>/models/<base>")
    p.add_argument("--deployment-id", required=True, help="short deployment id (no full path)")
    p.add_argument("--lora-models", required=True, help="comma-separated accounts/<acct>/models/<lora> list")
    p.add_argument("--eval-dataset", required=True, help="JSONL with prompt|messages per line")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--api-base", default="https://api.fireworks.ai")
    p.add_argument("--poll-interval-s", type=int, default=5)
    p.add_argument("--poll-timeout-s", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--deployment-shape",
        required=True,
        help="accounts/<acct>/deploymentShapes/<id>. Required — the shape owns the "
        "accelerator type/count and serving config, so no separate accelerator args.",
    )
    p.add_argument(
        "--no-create-deployment",
        action="store_true",
        help="reuse an existing deployment (implies --no-cleanup unless --cleanup is also given)",
    )
    p.add_argument(
        "--no-cleanup",
        action="store_true",
        help="leave the deployment running after eval (default on when --no-create-deployment is set)",
    )
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="force cleanup even when reusing an existing deployment with --no-create-deployment",
    )
    args = p.parse_args()

    if not args.account_id:
        sys.exit("--account-id required (or set FIREWORKS_ACCOUNT_NAME)")

    if args.no_cleanup and args.cleanup:
        sys.exit("--no-cleanup and --cleanup are mutually exclusive")
    # When reusing an existing deployment, default to NOT deleting it so a
    # deployment we didn't create is not removed. Opt back into cleanup with
    # --cleanup.
    if args.no_create_deployment:
        cleanup_deployment = bool(args.cleanup)
    else:
        cleanup_deployment = not args.no_cleanup

    return EvalConfig(
        account_id=args.account_id,
        base_model=args.base_model,
        deployment_id=args.deployment_id,
        lora_models=[m.strip() for m in args.lora_models.split(",") if m.strip()],
        eval_dataset_path=args.eval_dataset,
        results_dir=args.results_dir,
        api_base=args.api_base,
        poll_interval_s=args.poll_interval_s,
        poll_timeout_s=args.poll_timeout_s,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        min_replica_count=1,
        max_replica_count=1,
        deployment_shape=args.deployment_shape,
        create_deployment=not args.no_create_deployment,
        cleanup_deployment=cleanup_deployment,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    cfg = _parse_args()
    if not cfg.lora_models:
        sys.exit("--lora-models must list at least one LoRA")
    evaluator = MultiLoraHotReloadEvaluator(cfg)
    results = evaluator.run()
    summary = [
        {
            "lora": r.lora_model,
            "loaded_in_s": round(r.loaded_in_s, 1),
            "n_samples": r.n_samples,
            "n_errors": sum(1 for o in r.outputs if "error" in o),
            "error": r.error,
        }
        for r in results
    ]
    log.info("[summary]")
    for row in summary:
        log.info("  %s: loaded=%ss errors=%s/%s %s", row["lora"], row["loaded_in_s"], row["n_errors"], row["n_samples"], row["error"] or "")


if __name__ == "__main__":
    main()
