"""Build and run TITO inside one Harbor-managed agent environment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import shutil
import signal
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutSetup

SIDECAR_ROOT = "/tmp/fireworks-tito-sidecar"
SIDECAR_PYTHON = "/opt/fireworks-tito/bin/python"
SIDECAR_BUNDLE_ARCHIVE = f"{SIDECAR_ROOT}/bundle.zip"
SIDECAR_BUNDLE_ROOT = f"{SIDECAR_ROOT}/bundle"
SIDECAR_SPEC_PATH = f"{SIDECAR_ROOT}/spec.json"
SIDECAR_ENDPOINT_PATH = f"{SIDECAR_ROOT}/endpoint.json"
SIDECAR_TERMINAL_PATH = f"{SIDECAR_ROOT}/terminal.json"
SIDECAR_TERMINAL_STAGING_PATH = f"{SIDECAR_TERMINAL_PATH}.uploading"
SIDECAR_CONTEXT_OVERFLOW_PATH = f"{SIDECAR_ROOT}/context-budget-exhausted.json"
SIDECAR_PID_PATH = f"{SIDECAR_ROOT}/sidecar.pid"
SIDECAR_LOG_ROOT = "/logs/tito"
SIDECAR_ARTIFACT_PATH = f"{SIDECAR_LOG_ROOT}/trajectory.tito"
SIDECAR_ARTIFACT_MANIFEST_PATH = f"{SIDECAR_LOG_ROOT}/trajectory.json"
SIDECAR_DEBUG_ROOT = f"{SIDECAR_LOG_ROOT}/debug"
SIDECAR_COMPLETE_PATH = f"{SIDECAR_LOG_ROOT}/COMPLETE"
SIDECAR_STDOUT_PATH = f"{SIDECAR_LOG_ROOT}/sidecar.stdout"
SIDECAR_STDERR_PATH = f"{SIDECAR_LOG_ROOT}/sidecar.stderr"

_BUNDLE_VERSION = 6
# E2B can exhibit multi-minute startup outliers while the sidecar process is
# still alive. Keep readiness bounded by the wider agent-setup timeout, but do
# not kill a healthy process at the old two-minute bound observed under
# 64-way startup.
_SIDECAR_READY_TIMEOUT_SECONDS = 600
# Full-history long-context trajectories can produce compact artifacts hundreds
# of MiB in size.  Keep finalization bounded, but allow the same 15-minute
# origin window required by the sidecar transport contract; a 120-second bound
# killed a valid 500k-context artifact while it was being serialized.
_SIDECAR_ARTIFACT_FINALIZATION_TIMEOUT_SECONDS = 900
_DEBUG_MAX_LOCAL_BYTES = 2 * 1024**3
_DEBUG_MIN_FREE_BYTES = 512 * 1024**2
_IGNORED_NAMES = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", "tests"})
# This allowlist is the runtime counterpart of the training SDK package root's
# lazy-export contract. Every imported SDK dependency must be either listed here
# or contained in the copied ``sdk/tito`` tree; the extracted-bundle import test
# executes the real sidecar module to enforce that boundary.
_SDK_RUNTIME_FILES = (
    "__init__.py",
    "_rest_client.py",
    "_sse.py",
    "concurrency.py",
    "errors.py",
    "sampling.py",
    "sampling_observability.py",
    "tito_debug.py",
)
_COOKBOOK_RUNTIME_FILES = (
    "__init__.py",
    "examples/__init__.py",
    "examples/rl/__init__.py",
    "examples/rl/harbor/__init__.py",
    "examples/rl/harbor/tito/__init__.py",
    "examples/rl/harbor/tito/sidecar.py",
)
_CALL_CLASSIFIER_MODES = frozenset({"all_policy", "tools_present", "adapter_metadata"})


@dataclass(frozen=True)
class TITOSidecarBundle:
    path: Path
    digest: str


@dataclass(frozen=True)
class TITOSidecarLaunchSpec:
    schema_version: int
    bundle_digest: str
    inference_base_url: str
    api_key: str
    model: str
    renderer_name: str
    prompt_mode: str
    max_context_tokens: int
    max_output_tokens: int
    sampling_defaults: Mapping[str, Any]
    max_masked_tokens: int
    on_other_mismatch: str
    keepalive_seconds: float
    call_classifier: str
    trajectory_metadata: Mapping[str, Any]
    debug_enabled: bool
    debug_run_id: str | None
    debug_max_local_bytes: int | None
    debug_min_free_bytes: int | None
    debug_redact_text: bool


def _source_roots() -> tuple[Path, Path]:
    cookbook_root = Path(__file__).resolve().parents[5]
    sdk_module = __import__("fireworks.training.sdk", fromlist=["__file__"])
    sdk_file = getattr(sdk_module, "__file__", None)
    if not sdk_file:
        raise RuntimeError("could not resolve the imported training SDK package")
    sdk_source = Path(sdk_file).resolve().parent
    training_source = cookbook_root / "training"
    if not sdk_source.is_dir() or not training_source.is_dir():
        raise RuntimeError(
            "could not resolve SDK/Cookbook sources for the TITO sidecar bundle"
        )
    return sdk_source, training_source


def _iter_source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if _IGNORED_NAMES.intersection(path.parts):
            continue
        if path.is_symlink():
            raise ValueError(
                "TITO sidecar source bundle does not support symlinks: "
                f"{path.relative_to(root)}"
            )
        if not path.is_file():
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def _update_tree_hash(digest: Any, root: Path, prefix: str) -> None:
    for path in _iter_source_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{prefix}/{relative}\0".encode())
        digest.update(path.read_bytes())


def _copy_source_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "tests",
            "*.pyc",
            "*.pyo",
        ),
    )


def _copy_sdk_runtime(sdk_source: Path, destination: Path) -> None:
    """Copy only the lightweight training SDK needed by the sidecar."""
    sdk_target = destination / "fireworks" / "training" / "sdk"
    sdk_target.mkdir(parents=True)
    for name in _SDK_RUNTIME_FILES:
        shutil.copy2(sdk_source / name, sdk_target / name)
    _copy_source_tree(sdk_source / "tito", sdk_target / "tito")
    for package_root in (
        destination / "fireworks",
        destination / "fireworks" / "training",
    ):
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text(
            '"""Minimal package root for the immutable TITO sidecar bundle."""\n',
            encoding="utf-8",
        )


def _copy_cookbook_runtime(training_source: Path, destination: Path) -> None:
    training_target = destination / "training"
    _copy_source_tree(training_source / "tito", training_target / "tito")
    for name in _COOKBOOK_RUNTIME_FILES:
        relative = Path(name)
        source = training_source / relative
        target = training_target / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in _iter_source_files(source):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            archive.writestr(info, path.read_bytes(), compresslevel=6)


def build_sidecar_bundle(setup: RolloutSetup) -> TITOSidecarBundle:
    """Create one content-addressed source/tokenizer bundle shared by trials."""
    sdk_source, training_source = _source_roots()
    digest = hashlib.sha256()
    digest.update(f"tito-sidecar-bundle-v{_BUNDLE_VERSION}\0".encode())
    for name in _SDK_RUNTIME_FILES:
        source = sdk_source / name
        digest.update(f"python-sdk/fireworks/training/sdk/{name}\0".encode())
        digest.update(source.read_bytes())
    _update_tree_hash(
        digest,
        sdk_source / "tito",
        "python-sdk/fireworks/training/sdk/tito",
    )
    _update_tree_hash(digest, training_source / "tito", "cookbook/training/tito")
    for name in _COOKBOOK_RUNTIME_FILES:
        source = training_source / name
        digest.update(f"cookbook/training/{name}\0".encode())
        digest.update(source.read_bytes())
    tokenizer_backend = getattr(setup.tokenizer, "backend_tokenizer", None)
    if tokenizer_backend is None or not hasattr(tokenizer_backend, "to_str"):
        raise ValueError("TITO sidecar requires a serializable fast tokenizer")
    digest.update(tokenizer_backend.to_str().encode())
    digest.update(
        json.dumps(setup.tokenizer.special_tokens_map, sort_keys=True).encode()
    )
    chat_template = getattr(setup.tokenizer, "chat_template", None)
    if not chat_template:
        raise ValueError("TITO sidecar requires a tokenizer chat template")
    digest.update(
        json.dumps(chat_template, sort_keys=True, ensure_ascii=False).encode()
    )
    bundle_digest = digest.hexdigest()

    configured_root = setup.extras.get("tito_sidecar_bundle_root")
    if configured_root is None:
        trials_root = setup.extras.get("harbor_trials_dir")
        if trials_root is None:
            raise ValueError(
                "rollout_extras must set harbor_trials_dir or tito_sidecar_bundle_root; "
                "the sidecar bundle must live on persistent rollout storage"
            )
        configured_root = (
            Path(trials_root).expanduser().resolve().parent / ".tito-sidecar-bundles"
        )
    bundle_root = Path(configured_root).expanduser().resolve()
    destination = bundle_root / f"{bundle_digest}.zip"
    if destination.is_file():
        return TITOSidecarBundle(destination, bundle_digest)

    bundle_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_digest}.", dir=bundle_root))
    descriptor, raw_archive_path = tempfile.mkstemp(
        prefix=f".{bundle_digest}.",
        suffix=".zip",
        dir=bundle_root,
    )
    os.close(descriptor)
    temporary_archive = Path(raw_archive_path)
    try:
        _copy_sdk_runtime(sdk_source, temporary / "python-sdk")
        _copy_cookbook_runtime(training_source, temporary / "cookbook")
        tokenizer_dir = temporary / "tokenizer"
        setup.tokenizer.save_pretrained(tokenizer_dir)
        manifest = {
            "schema_version": _BUNDLE_VERSION,
            "sha256": bundle_digest,
            "tokenizer_id": setup.tokenizer_id,
            "model": setup.model,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        _write_deterministic_zip(temporary, temporary_archive)
        temporary_archive.replace(destination)
    except BaseException:
        temporary_archive.unlink(missing_ok=True)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return TITOSidecarBundle(destination, bundle_digest)


def build_launch_spec(
    setup: RolloutSetup,
    bundle: TITOSidecarBundle,
    *,
    call_classifier: str,
    metadata: Mapping[str, Any],
) -> TITOSidecarLaunchSpec:
    if call_classifier not in _CALL_CLASSIFIER_MODES:
        raise ValueError(f"unsupported TITO call classifier: {call_classifier}")
    renderer_name = str(setup.extras.get("renderer_name") or "")
    if not renderer_name:
        raise ValueError("rollout_extras['renderer_name'] is required")
    max_context_tokens = resolve_max_context_tokens(setup)
    max_output_tokens = int(setup.sample_kwargs.get("max_tokens") or 0)
    if (
        max_context_tokens < 2
        or max_output_tokens < 1
        or max_output_tokens >= max_context_tokens
    ):
        raise ValueError("invalid sidecar context/output token limits")
    sampling_defaults = {
        key: value
        for key, value in setup.sample_kwargs.items()
        if key not in {"max_tokens", "max_seq_len", "echo"}
    }
    debug_enabled = bool(setup.extras.get("tito_debug_enabled", False))
    max_masked_tokens = int(setup.extras.get("tito_max_masked_tokens", 1024))
    if max_masked_tokens < 0:
        raise ValueError(
            "rollout_extras['tito_max_masked_tokens'] must be non-negative"
        )
    on_other_mismatch = str(setup.extras.get("tito_on_other_mismatch", "new_segment"))
    if on_other_mismatch not in {"new_segment", "reject"}:
        raise ValueError("invalid TITO drift policy")
    prompt_mode = str(setup.extras.get("tito_prompt_mode", "full_history"))
    if prompt_mode not in {"full_history", "incremental"}:
        raise ValueError("invalid TITO prompt mode")
    keepalive_seconds = float(setup.extras.get("tito_keepalive_seconds", 5.0))
    if keepalive_seconds <= 0:
        raise ValueError("rollout_extras['tito_keepalive_seconds'] must be positive")
    return TITOSidecarLaunchSpec(
        schema_version=2,
        bundle_digest=bundle.digest,
        inference_base_url=setup.inference_base_url,
        api_key=setup.api_key,
        model=setup.model,
        renderer_name=renderer_name,
        prompt_mode=prompt_mode,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        sampling_defaults=sampling_defaults,
        max_masked_tokens=max_masked_tokens,
        on_other_mismatch=on_other_mismatch,
        keepalive_seconds=keepalive_seconds,
        call_classifier=call_classifier,
        trajectory_metadata=dict(metadata),
        debug_enabled=debug_enabled,
        debug_run_id=None,
        debug_max_local_bytes=_DEBUG_MAX_LOCAL_BYTES if debug_enabled else None,
        debug_min_free_bytes=_DEBUG_MIN_FREE_BYTES if debug_enabled else None,
        debug_redact_text=bool(setup.extras.get("tito_debug_redact_text", False)),
    )


def resolve_max_context_tokens(setup: RolloutSetup) -> int:
    """Translate the rollout's single ``max_seq_len`` into TITO terminology."""

    max_context_tokens = int(setup.sample_kwargs.get("max_seq_len") or 0)
    if max_context_tokens < 1:
        raise ValueError("sample_kwargs['max_seq_len'] must be positive")
    return max_context_tokens


def launch_spec_json(spec: TITOSidecarLaunchSpec) -> str:
    return json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))


def _temporary_private_file(payload: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix="tito-sidecar-", suffix=".json")
    path = Path(raw_path)
    try:
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o600)
    finally:
        # ``mkstemp`` returns an open descriptor; Path.write_text opens its own
        # descriptor and does not close this one.
        os.close(descriptor)
    return path


async def upload_private_text(
    environment: Any,
    *,
    content: str,
    remote_path: str,
) -> None:
    """Upload generated private configuration without logging its contents."""

    local_path = _temporary_private_file(content)
    try:
        await environment.upload_file(local_path, remote_path)
    finally:
        local_path.unlink(missing_ok=True)


async def install_sidecar(
    environment: Any,
    *,
    bundle_path: str | Path,
    launch_spec: str,
) -> dict[str, str]:
    """Upload and start one environment-local sidecar, then return its endpoint."""

    spec_path = _temporary_private_file(launch_spec)
    try:
        await environment.exec(
            command=(
                f"mkdir -p {SIDECAR_ROOT} {SIDECAR_LOG_ROOT} && "
                f"chmod 700 {SIDECAR_ROOT} {SIDECAR_LOG_ROOT}"
            ),
            cwd="/",
        )
        await environment.upload_file(Path(bundle_path), SIDECAR_BUNDLE_ARCHIVE)
        await environment.upload_file(spec_path, SIDECAR_SPEC_PATH)
    finally:
        spec_path.unlink(missing_ok=True)

    started = await environment.exec(
        command=(
            "set -eu; "
            f"chmod 600 {SIDECAR_SPEC_PATH}; "
            f"rm -rf {SIDECAR_BUNDLE_ROOT}; mkdir -p {SIDECAR_BUNDLE_ROOT}; "
            f"{SIDECAR_PYTHON} -m zipfile -e {SIDECAR_BUNDLE_ARCHIVE} {SIDECAR_BUNDLE_ROOT}; "
            f"PYTHONPATH={SIDECAR_BUNDLE_ROOT}/python-sdk:"
            f"{SIDECAR_BUNDLE_ROOT}/cookbook "
            f"nohup {SIDECAR_PYTHON} -m training.examples.rl.harbor.tito.sidecar serve "
            f"--spec {SIDECAR_SPEC_PATH} "
            f">{SIDECAR_LOG_ROOT}/sidecar.stdout 2>"
            f"{SIDECAR_LOG_ROOT}/sidecar.stderr </dev/null & "
            f"sidecar_pid=$!; printf '%s\n' \"$sidecar_pid\" > {SIDECAR_PID_PATH}"
        ),
        cwd="/",
    )
    if started.return_code != 0:
        raise RuntimeError("failed to start the TITO sidecar process")

    deadline = asyncio.get_running_loop().time() + _SIDECAR_READY_TIMEOUT_SECONDS
    while True:
        result = await environment.exec(
            command=(
                f"if test -s {SIDECAR_ENDPOINT_PATH}; then "
                f"cat {SIDECAR_ENDPOINT_PATH}; "
                f"elif test -s {SIDECAR_PID_PATH} && "
                f'kill -0 "$(cat {SIDECAR_PID_PATH})" 2>/dev/null; then exit 2; '
                f"else tail -c 8192 {SIDECAR_LOG_ROOT}/sidecar.stderr 2>/dev/null; exit 3; fi"
            ),
            cwd="/",
        )
        if result.return_code == 0:
            endpoint = json.loads(result.stdout or "{}")
            if not isinstance(endpoint, dict):
                raise RuntimeError("TITO sidecar returned a non-object endpoint")
            required = {"trajectory_id", "openai_base_url", "api_key"}
            if not required.issubset(endpoint):
                raise RuntimeError("TITO sidecar endpoint is missing required fields")
            return {name: str(endpoint[name]) for name in required}
        if result.return_code != 2:
            detail = (result.stdout or result.stderr or "").strip()
            raise RuntimeError(
                f"TITO sidecar exited before readiness: {detail[-4096:]}"
            )
        if asyncio.get_running_loop().time() >= deadline:
            await stop_sidecar_process(environment)
            raise TimeoutError(
                f"TITO sidecar did not become ready within {_SIDECAR_READY_TIMEOUT_SECONDS}s"
            )
        await asyncio.sleep(1)


async def stop_sidecar_process(environment: Any) -> None:
    """Best-effort process cleanup for a sidecar that cannot terminalize."""

    await environment.exec(
        command=(
            f"if test -s {SIDECAR_PID_PATH}; then "
            f"sidecar_pid=$(cat {SIDECAR_PID_PATH}); "
            'kill "$sidecar_pid" 2>/dev/null || true; '
            "fi"
        ),
        cwd="/",
    )


async def quiesce_harness_process(
    environment: Any,
    *,
    process_pattern: str,
) -> None:
    """Stop an orphaned in-sandbox harness before terminalizing TITO state.

    Cancelling Harbor's host-side ``docker compose exec`` does not guarantee
    that the command inside the container exits.  If that process keeps using
    the loopback endpoint while the sidecar is being abandoned, terminal
    artifact publication can race or hang.  Each adapter supplies a narrow
    process signature for its own CLI; the bracketed signature also avoids
    matching this cleanup command's ``pgrep`` invocation.
    """

    if not process_pattern or "\n" in process_pattern:
        raise ValueError("process_pattern must be a non-empty single line")
    result = await environment.exec(
        command=(
            "if command -v pgrep >/dev/null 2>&1; then "
            f"pattern={shlex.quote(process_pattern)}; "
            'pids=$(pgrep -f -- "$pattern" || true); '
            'if test -n "$pids"; then '
            "kill -TERM $pids 2>/dev/null || true; "
            "for _attempt in {1..50}; do "
            'pgrep -f -- "$pattern" >/dev/null 2>&1 || break; '
            "sleep 0.1; "
            "done; "
            'pids=$(pgrep -f -- "$pattern" || true); '
            'test -z "$pids" || kill -KILL $pids 2>/dev/null || true; '
            "fi; "
            "fi"
        ),
        cwd="/",
    )
    if result.return_code != 0:
        detail = (result.stdout or result.stderr or "").strip()
        raise RuntimeError(f"failed to stop the sandbox harness: {detail[-4096:]}")


async def abandon_sidecar_after_harness_cancellation(
    environment: Any,
    *,
    process_pattern: str,
) -> None:
    """Quiesce the cancelled harness, then durably abandon its trajectory."""

    await quiesce_harness_process(
        environment,
        process_pattern=process_pattern,
    )
    await terminalize_sidecar(
        environment,
        status="abandoned",
        reason="agent_cancelled",
    )


async def terminalize_sidecar(
    environment: Any,
    *,
    status: str,
    reason: str | None = None,
) -> None:
    """Publish terminal intent and wait until the artifact is durable."""

    if status not in {"completed", "abandoned", "failed"}:
        raise ValueError(f"invalid TITO sidecar terminal status: {status!r}")
    payload = json.dumps({"status": status, "reason": reason}, sort_keys=True)
    terminal_path = _temporary_private_file(payload)
    try:
        # Remote-provider uploads are not guaranteed to appear atomically.  The
        # sidecar polls this file, so publish through a sibling path and rename
        # only after the complete JSON payload is durable in the sandbox.
        await environment.upload_file(terminal_path, SIDECAR_TERMINAL_STAGING_PATH)
        published = await environment.exec(
            command=(
                f"chmod 600 {SIDECAR_TERMINAL_STAGING_PATH} && "
                f"mv -f {SIDECAR_TERMINAL_STAGING_PATH} {SIDECAR_TERMINAL_PATH}"
            ),
            cwd="/",
        )
        if published.return_code != 0:
            detail = (published.stdout or published.stderr or "").strip()
            raise RuntimeError(
                f"failed to publish TITO sidecar terminal intent: {detail[-4096:]}"
            )
    finally:
        terminal_path.unlink(missing_ok=True)

    deadline = (
        asyncio.get_running_loop().time()
        + _SIDECAR_ARTIFACT_FINALIZATION_TIMEOUT_SECONDS
    )
    while True:
        result = await environment.exec(
            command=(
                f"if test -s {SIDECAR_COMPLETE_PATH}; then exit 0; "
                f"elif test -s {SIDECAR_PID_PATH} && "
                f'kill -0 "$(cat {SIDECAR_PID_PATH})" 2>/dev/null; then exit 2; '
                f"else tail -c 8192 {SIDECAR_LOG_ROOT}/sidecar.stderr 2>/dev/null; exit 3; fi"
            ),
            cwd="/",
        )
        if result.return_code == 0:
            return
        if result.return_code != 2:
            detail = (result.stdout or result.stderr or "").strip()
            raise RuntimeError(
                "TITO sidecar exited before publishing its artifact: " + detail[-4096:]
            )
        if asyncio.get_running_loop().time() >= deadline:
            await stop_sidecar_process(environment)
            raise TimeoutError(
                "TITO sidecar did not publish its terminal artifact within "
                f"{_SIDECAR_ARTIFACT_FINALIZATION_TIMEOUT_SECONDS}s"
            )
        await asyncio.sleep(1)


def build_call_classifier(mode: str):
    """Build one protocol-level classifier selected by a harness adapter."""

    from fireworks.training.sdk import TITOChatRequest

    if mode == "tools_present":
        return lambda request: (
            ("policy", "tools_present")
            if request.tools
            else ("auxiliary", "tools_absent")
        )
    if mode == "all_policy":
        return lambda request: ("policy", "all_policy")
    if mode == "adapter_metadata":

        def classify_metadata(request: TITOChatRequest) -> tuple[str, str]:
            kind = request.adapter_metadata.get("call_kind")
            source = request.adapter_metadata.get("classifier_source")
            if kind in {"policy", "auxiliary"}:
                return str(kind), str(source or "adapter_metadata")
            raise ValueError("request is missing a valid _tito.call_kind")

        return classify_metadata
    raise ValueError(f"unsupported TITO call classifier: {mode}")


def _atomic_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.chmod(mode)
    temporary.replace(path)


async def sidecar_failure_disposition(environment: Any) -> str | None:
    """Read the sidecar's typed last-policy disposition without log matching."""

    result = await environment.exec(
        command=(
            f"if test -s {SIDECAR_CONTEXT_OVERFLOW_PATH}; then "
            "printf context_budget_exhausted; fi"
        ),
        cwd="/",
    )
    value = str(result.stdout or "").strip()
    if value == "context_budget_exhausted":
        return value
    return None


async def _wait_for_terminal(
    path: Path, interrupted: asyncio.Event
) -> tuple[str, str | None]:
    while not interrupted.is_set():
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            status = str(raw.get("status") or "")
            reason = raw.get("reason")
            if status not in {"completed", "abandoned", "failed"}:
                raise ValueError(f"invalid sidecar terminal status: {status!r}")
            return status, None if reason is None else str(reason)
        await asyncio.sleep(0.1)
    return "abandoned", "sidecar_process_interrupted"


async def serve(spec_path: Path) -> None:
    from fireworks.training.sdk import (
        DeploymentSampler,
        TITOLocalDebugConfig,
        TITOLocalDebugSink,
        TITOSidecar,
        TrajectoryDriftPolicy,
    )
    from training.tito.renderer import (
        build_sidecar_tito_renderer,
        load_sidecar_tokenizer,
    )

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 2:
        raise ValueError("unsupported sidecar launch spec")
    bundle_manifest = json.loads(
        (Path(SIDECAR_BUNDLE_ROOT) / "manifest.json").read_text(encoding="utf-8")
    )
    if bundle_manifest.get("sha256") != spec.get("bundle_digest"):
        raise ValueError("sidecar bundle digest does not match launch spec")
    tokenizer = load_sidecar_tokenizer(Path(SIDECAR_BUNDLE_ROOT) / "tokenizer")
    renderer = build_sidecar_tito_renderer(
        tokenizer,
        str(spec["renderer_name"]),
    )
    sampler = DeploymentSampler(
        inference_url=str(spec["inference_base_url"]),
        model=str(spec["model"]),
        api_key=str(spec["api_key"]),
        tokenizer=tokenizer,
    )
    observer = None
    if bool(spec.get("debug_enabled")):
        observer = TITOLocalDebugSink(
            TITOLocalDebugConfig(
                root_dir=Path(SIDECAR_DEBUG_ROOT),
                run_id=spec.get("debug_run_id"),
                max_local_bytes=int(spec["debug_max_local_bytes"]),
                min_free_bytes=int(spec["debug_min_free_bytes"]),
                redact_text=bool(spec.get("debug_redact_text", False)),
                metadata={"runtime": "harbor-tito-sidecar"},
            )
        )
    call_classifier = build_call_classifier(str(spec["call_classifier"]))

    class _ContextAwareSidecar(TITOSidecar):
        """Expose context exhaustion to the harness as a typed terminal reason."""

        async def _handle_openai_chat(self, request: Any) -> Any:
            response = await super()._handle_openai_chat(request)
            error_code: str | None = None
            body = getattr(response, "body", None)
            if body:
                try:
                    error_code = str(
                        (json.loads(bytes(body)).get("error") or {}).get("code") or ""
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    error_code = None
            marker = Path(SIDECAR_CONTEXT_OVERFLOW_PATH)
            if error_code == "tito_context_overflow":
                _atomic_write(
                    marker,
                    json.dumps(
                        {"reason": "context_budget_exhausted"}, sort_keys=True
                    ).encode()
                    + b"\n",
                )
            else:
                marker.unlink(missing_ok=True)
            return response

    sidecar = _ContextAwareSidecar.from_deployment_sampler(
        sampler,
        renderer=renderer,
        max_context_tokens=int(spec["max_context_tokens"]),
        max_output_tokens=int(spec["max_output_tokens"]),
        call_classifier=call_classifier,
        sampling_defaults=dict(spec.get("sampling_defaults") or {}),
        observer=observer,
        keepalive_seconds=float(spec.get("keepalive_seconds", 5.0)),
        default_drift_policy=TrajectoryDriftPolicy(
            max_masked_tokens=int(spec.get("max_masked_tokens", 1024)),
            on_other_mismatch=str(spec.get("on_other_mismatch", "new_segment")),  # type: ignore[arg-type]
        ),
        prompt_mode=str(spec.get("prompt_mode", "full_history")),  # type: ignore[arg-type]
    )
    try:
        interrupted = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, interrupted.set)

        await sidecar.start()
        _atomic_write(Path(SIDECAR_PID_PATH), f"{os.getpid()}\n".encode())
        endpoint = await sidecar.create_trajectory_async(
            metadata=dict(spec.get("trajectory_metadata") or {})
        )
        _atomic_write(
            Path(SIDECAR_ENDPOINT_PATH),
            json.dumps(asdict(endpoint), sort_keys=True).encode() + b"\n",
        )
        agent_started = time.monotonic()
        status, reason = await _wait_for_terminal(
            Path(SIDECAR_TERMINAL_PATH), interrupted
        )
        await sidecar.observe_agent_wall(
            endpoint.trajectory_id,
            time.monotonic() - agent_started,
        )
        if status == "completed":
            artifact = await sidecar.finish_trajectory(endpoint.trajectory_id)
        elif status == "failed":
            artifact = await sidecar.fail_trajectory(
                endpoint.trajectory_id, reason or "agent_failed"
            )
        else:
            artifact = await sidecar.abandon_trajectory(
                endpoint.trajectory_id, reason or "agent_abandoned"
            )
        encoded = artifact.pack()
        _atomic_write(Path(SIDECAR_ARTIFACT_PATH), encoded)
        _atomic_write(
            Path(SIDECAR_ARTIFACT_MANIFEST_PATH),
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "trajectory_id": artifact.trajectory_id,
                        "status": artifact.status,
                        "terminal_reason": artifact.terminal_reason,
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "bytes": len(encoded),
                        "bundle_digest": spec["bundle_digest"],
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode(),
        )
        _atomic_write(Path(SIDECAR_COMPLETE_PATH), b"complete\n")
    finally:
        try:
            await sidecar.close()
        finally:
            sampler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve", nargs="?")
    parser.add_argument("--spec", type=Path, default=Path(SIDECAR_SPEC_PATH))
    args = parser.parse_args()
    if args.serve != "serve":
        parser.error("expected the 'serve' command")
    asyncio.run(serve(args.spec))


if __name__ == "__main__":
    main()


__all__ = [
    "SIDECAR_ARTIFACT_MANIFEST_PATH",
    "SIDECAR_ARTIFACT_PATH",
    "SIDECAR_BUNDLE_ARCHIVE",
    "SIDECAR_BUNDLE_ROOT",
    "SIDECAR_COMPLETE_PATH",
    "SIDECAR_CONTEXT_OVERFLOW_PATH",
    "SIDECAR_DEBUG_ROOT",
    "SIDECAR_ENDPOINT_PATH",
    "SIDECAR_LOG_ROOT",
    "SIDECAR_STDERR_PATH",
    "SIDECAR_STDOUT_PATH",
    "SIDECAR_PID_PATH",
    "SIDECAR_PYTHON",
    "SIDECAR_ROOT",
    "SIDECAR_SPEC_PATH",
    "SIDECAR_TERMINAL_PATH",
    "TITOSidecarBundle",
    "TITOSidecarLaunchSpec",
    "abandon_sidecar_after_harness_cancellation",
    "build_launch_spec",
    "build_call_classifier",
    "build_sidecar_bundle",
    "install_sidecar",
    "launch_spec_json",
    "resolve_max_context_tokens",
    "sidecar_failure_disposition",
    "stop_sidecar_process",
    "terminalize_sidecar",
    "upload_private_text",
]
