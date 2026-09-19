"""Generate and validate deterministic TextWorld Harbor task suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEXTWORLD_VERSION = "1.6.2"
GENERATOR_NUMPY_VERSION = "1.26.4"
MANIFEST_NAME = "textworld-manifest.json"
MAX_GAME_SEED_ATTEMPTS = 64

COOKING = "cooking"
_COOKING_FLAGS = ("open", "cook", "cut", "drop")
CALIBRATED_COOKING_SETTINGS: dict[str, Any] = {
    "recipe": 5,
    "take": 5,
    "go": 12,
    "open": True,
    "cook": True,
    "cut": True,
    "drop": True,
    "split": "test",
}


_CLI = r'''#!/usr/bin/env python3
"""Persistent shell interface for the frozen TextWorld game."""

import json
import os
import sys
import tempfile
from pathlib import Path

import textworld

GAME = os.environ.get("TEXTWORLD_GAME", "/opt/textworld/game.ulx")
ACTIONS = Path(
    os.environ.get("TEXTWORLD_ACTIONS", "/workspace/.textworld-actions.json")
)
MAX_ACTIONS = 1000


def load_actions():
    if not ACTIONS.exists():
        return []
    value = json.loads(ACTIONS.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("action history is malformed")
    if len(value) > MAX_ACTIONS:
        raise ValueError("action history is too long")
    return value


def save_actions(actions):
    fd, temporary = tempfile.mkstemp(dir=ACTIONS.parent, prefix=".textworld-actions-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(actions, stream)
            stream.write("\n")
        os.replace(temporary, ACTIONS)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def start():
    # The oracle-bearing game.json sidecar is verifier-only. The compiled
    # interpreter still provides all observations needed by the agent.
    environment = textworld.start(GAME)
    return environment, environment.reset()


def finished(state):
    feedback = state.feedback or ""
    return "*** The End ***" in feedback or "*** You lost! ***" in feedback


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: textworld ACTION (example: textworld "go north")')
    command = " ".join(sys.argv[1:]).strip()
    if not command:
        raise SystemExit("TextWorld action must not be empty")

    actions = load_actions()
    environment, state = start()
    # The reset feedback carries the goal statement (for cooking, the pointer
    # to the cookbook). Show it once, on the first action of the episode.
    if not actions:
        print(state.feedback)
    done = False
    for previous in actions:
        state, _, done = environment.step(previous)
        if done or finished(state):
            break
    if done or finished(state):
        print(state.feedback)
        print("[finished=True]")
        return
    if len(actions) >= MAX_ACTIONS:
        raise SystemExit(f"maximum action count ({MAX_ACTIONS}) reached")

    state, _, done = environment.step(command)
    actions.append(command)
    save_actions(actions)
    print(state.feedback)
    print(f"[finished={done or finished(state)}]")
    environment.close()


if __name__ == "__main__":
    main()
'''

_RESET = r'''#!/bin/sh
set -eu
rm -f /workspace/.textworld-actions.json
exec textworld look
'''

_GRADER = r'''#!/usr/bin/env python3
"""Replay the submitted actions and emit a binary TextWorld reward."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import textworld

GAME = os.environ.get("TEXTWORLD_GAME", "/opt/textworld/game.ulx")
ACTIONS = Path(
    os.environ.get("TEXTWORLD_ACTIONS", "/workspace/.textworld-actions.json")
)
REWARD = Path(
    os.environ.get("TEXTWORLD_REWARD", "/logs/verifier/reward.txt")
)
METADATA = Path(os.environ.get("TEXTWORLD_METADATA", "/tests/game.json"))


def grade():
    actions = json.loads(ACTIONS.read_text(encoding="utf-8"))
    if not isinstance(actions, list) or not actions or len(actions) > 1000:
        return 0.0
    if any(not isinstance(action, str) or not action.strip() for action in actions):
        return 0.0
    with tempfile.TemporaryDirectory(prefix="textworld-verifier-") as directory:
        game = Path(directory) / "game.ulx"
        shutil.copyfile(GAME, game)
        shutil.copyfile(METADATA, game.with_suffix(".json"))
        infos = textworld.EnvInfos(won=True, lost=True)
        environment = textworld.start(str(game), infos)
        state = environment.reset()
        done = False
        for action in actions:
            state, _, done = environment.step(action)
            if done:
                break
        environment.close()
        return 1.0 if done and bool(state.won) else 0.0


try:
    reward = grade()
except Exception:
    reward = 0.0
REWARD.parent.mkdir(parents=True, exist_ok=True)
REWARD.write_text(f"{reward:g}\n", encoding="utf-8")
'''


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        if path.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + relative + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def ensure_generator_environment(root: str | Path) -> Path:
    """Create or verify the pinned TextWorld generation environment.

    TextWorld 1.6.2's cooking generator relies on NumPy 1.x scalar coercion,
    while the task image and TITO sidecar require newer runtime dependencies.
    Keep generation isolated instead of constraining the training environment.
    """

    environment = Path(root).expanduser().resolve()
    python = environment / "bin" / "python"
    tw_make = environment / "bin" / "tw-make"
    check = (
        "import numpy, textworld; "
        f"assert textworld.__version__ == {TEXTWORLD_VERSION!r}; "
        f"assert numpy.__version__ == {GENERATOR_NUMPY_VERSION!r}"
    )
    if python.is_file() and tw_make.is_file():
        result = subprocess.run([str(python), "-c", check], check=False)
        if result.returncode:
            raise RuntimeError(
                f"generator environment has incompatible packages: {environment}"
            )
        return tw_make
    if environment.exists():
        raise RuntimeError(
            f"generator environment is incomplete: {environment}; remove it and retry"
        )

    environment.parent.mkdir(parents=True, exist_ok=True)
    try:
        venv.EnvBuilder(with_pip=True).create(environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                f"textworld=={TEXTWORLD_VERSION}",
                f"numpy=={GENERATOR_NUMPY_VERSION}",
            ],
            check=True,
        )
        subprocess.run([str(python), "-c", check], check=True)
    except BaseException:
        shutil.rmtree(environment, ignore_errors=True)
        raise
    return tw_make


def _task_toml(
    *,
    task_id: str,
    split: str,
    seed: int,
    settings: dict[str, Any],
) -> str:
    return f'''schema_version = "1.4"

[task]
name = "{task_id}"
version = "1.0.0"
description = "Solve a frozen TextWorld cooking game"
authors = [{{ name = "Fireworks AI" }}]
keywords = ["textworld", "interactive-fiction", "reinforcement-learning"]

[metadata]
dataset = "textworld-cooking"
challenge = "cooking"
split = "{split}"
seed = {seed}
challenge_settings = '{json.dumps(settings, sort_keys=True)}'
textworld_version = "{TEXTWORLD_VERSION}"

[agent]
timeout_sec = 1800.0

[verifier]
timeout_sec = 300.0

[environment]
build_timeout_sec = 1800.0
cpus = 2
memory_mb = 4096
storage_mb = 8192
gpus = 0
'''


def _dockerfile() -> str:
    # The shared TITO sidecar layer pins numpy, which requires Python >= 3.11.
    return f'''FROM python:3.12-slim-bookworm

RUN apt-get update \\
 && apt-get install -y --no-install-recommends build-essential libffi-dev python3-dev curl git \\
 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir textworld=={TEXTWORLD_VERSION}
RUN useradd --create-home --shell /bin/bash agent \\
 && mkdir -p /opt/textworld /workspace \\
 && chown agent:agent /workspace
COPY game.ulx /opt/textworld/game.ulx
COPY textworld_cli.py /usr/local/bin/textworld
COPY textworld_reset.sh /usr/local/bin/textworld-reset
RUN chmod 0755 /usr/local/bin/textworld /usr/local/bin/textworld-reset
USER agent
WORKDIR /workspace
'''


_BASE_INSTRUCTION = (
    "Solve the TextWorld game. Use `textworld_action` for exactly one game "
    'action at a time (for example `"look"` or `"go north"`). Use '
    "`textworld_reset` only when restarting the game is necessary. "
    "The game prints its own goal on your first action. "
    "The task succeeds only when the game finishes in a won state.\n"
)


def _cooking_argv(settings: dict[str, Any], seed: int) -> list[str]:
    """Build the tw-make command for one cooking configuration."""
    argv = [
        "tw-cooking",
        "--recipe",
        str(settings["recipe"]),
        "--take",
        str(settings["take"]),
        "--go",
        str(settings["go"]),
        # Vary the recipe per game; a fixed recipe seed repeats menus.
        "--recipe-seed",
        str(seed),
    ]
    for flag in _COOKING_FLAGS:
        if settings.get(flag):
            argv.append(f"--{flag}")
    if settings.get("split"):
        argv.extend(["--split", str(settings["split"])])
    return argv


def _create_task(
    root: Path,
    *,
    split: str,
    index: int,
    seed: int,
    settings: dict[str, Any],
    tw_make: str,
) -> tuple[str, str]:
    directory_name = f"textworld-{split}-{index:05d}"
    task_id = f"fireworks/{directory_name}"
    task = root / directory_name
    environment = task / "environment"
    environment.mkdir(parents=True)
    game = environment / "game.ulx"
    subprocess.run(
        [
            tw_make,
            *_cooking_argv(settings, seed),
            "--seed",
            str(seed),
            "--format",
            "ulx",
            "--output",
            str(game),
            "--silent",
        ],
        check=True,
    )
    if not game.is_file() or game.stat().st_size == 0:
        raise RuntimeError(f"tw-make did not produce {game}")
    game_metadata = game.with_suffix(".json")
    if not game_metadata.is_file():
        raise RuntimeError(f"tw-make did not produce {game_metadata}")

    _write(
        task / "task.toml",
        _task_toml(
            task_id=task_id,
            split=split,
            seed=seed,
            settings=settings,
        ),
    )
    _write(task / "instruction.md", _BASE_INSTRUCTION)
    _write(environment / "Dockerfile", _dockerfile())
    _write(environment / "textworld_cli.py", _CLI, executable=True)
    _write(environment / "textworld_reset.sh", _RESET, executable=True)
    (task / "tests").mkdir(parents=True, exist_ok=True)
    shutil.move(game_metadata, task / "tests" / "game.json")
    game.with_suffix(".ni").unlink(missing_ok=True)
    _write(task / "tests" / "grader.py", _GRADER)
    _write(
        task / "tests" / "test.sh",
        "#!/bin/sh\nset -eu\npython /tests/grader.py\n",
        executable=True,
    )
    return directory_name, task_id


@dataclass(frozen=True, slots=True)
class FrozenTextWorldDataset:
    root: Path
    train_task_ids: tuple[str, ...]
    evaluation_task_ids: tuple[str, ...]
    directory_by_task_id: dict[str, str]
    content_sha256: dict[str, str]
    manifest_sha256: str


def generate_dataset(
    output: str | Path,
    *,
    seed: int,
    train_tasks: int,
    evaluation_tasks: int,
    settings: dict[str, Any] | None = None,
    tw_make: str = "tw-make",
) -> FrozenTextWorldDataset:
    """Generate, freeze, and hash one TextWorld cooking suite."""
    if train_tasks < 1 or evaluation_tasks < 1:
        raise ValueError("train_tasks and evaluation_tasks must both be positive")
    settings = dict(CALIBRATED_COOKING_SETTINGS if settings is None else settings)
    recipe = int(settings.get("recipe", 0))
    take = int(settings.get("take", 0))
    if not 1 <= recipe <= 5:
        raise ValueError("cooking recipe must have between 1 and 5 ingredients")
    if not 0 <= take <= recipe:
        raise ValueError("cooking take must be between 0 and recipe")
    if int(settings.get("go", 1)) not in (1, 6, 9, 12):
        raise ValueError("cooking go must be one of 1, 6, 9, or 12")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if shutil.which(tw_make) is None:
        raise RuntimeError(
            f"{tw_make!r} is unavailable; install textworld=={TEXTWORLD_VERSION}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        rng = random.Random(seed)
        records: list[dict[str, Any]] = []
        for split, count in (("train", train_tasks), ("eval", evaluation_tasks)):
            for index in range(count):
                directory = ""
                registry_name = ""
                game_seed = 0
                for attempt in range(MAX_GAME_SEED_ATTEMPTS):
                    game_seed = rng.randrange(1, 2**31)
                    try:
                        directory, registry_name = _create_task(
                            temporary,
                            split=split,
                            index=index,
                            seed=game_seed,
                            settings=settings,
                            tw_make=tw_make,
                        )
                        break
                    except subprocess.CalledProcessError:
                        shutil.rmtree(
                            temporary / f"textworld-{split}-{index:05d}",
                            ignore_errors=True,
                        )
                else:
                    raise RuntimeError(
                        f"could not compile TextWorld {split} task {index} after "
                        f"{MAX_GAME_SEED_ATTEMPTS} seeds for cooking {settings}"
                    )
                records.append(
                    {
                        "task_id": directory,
                        "registry_name": registry_name,
                        "directory": directory,
                        "split": split,
                        "seed": game_seed,
                    }
                )
        content_sha256 = {
            record["task_id"]: _tree_sha256(temporary / record["directory"])
            for record in records
        }
        manifest = {
            "schema_version": 1,
            "challenge": COOKING,
            "settings": settings,
            "textworld_version": TEXTWORLD_VERSION,
            "dataset_seed": seed,
            "tasks": records,
            "content_sha256": content_sha256,
        }
        _write(
            temporary / MANIFEST_NAME,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_frozen_dataset(destination)


def load_frozen_dataset(root: str | Path) -> FrozenTextWorldDataset:
    """Load a generated suite and reject membership or content drift."""
    dataset_root = Path(root).expanduser().resolve()
    manifest_path = dataset_root / MANIFEST_NAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("textworld_version") != TEXTWORLD_VERSION:
        raise ValueError("TextWorld dataset version does not match the recipe")
    records = list(document.get("tasks") or ())
    task_ids = [str(record.get("task_id")) for record in records]
    directories = [str(record.get("directory")) for record in records]
    if not records or len(task_ids) != len(set(task_ids)) or len(directories) != len(set(directories)):
        raise ValueError("TextWorld manifest has missing or duplicate tasks")
    actual_directories = {
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "task.toml").is_file()
    }
    if actual_directories != set(directories):
        raise ValueError("TextWorld task membership differs from the manifest")
    expected_hashes = dict(document.get("content_sha256") or {})
    actual_hashes = {
        task_id: _tree_sha256(dataset_root / directory)
        for task_id, directory in zip(task_ids, directories, strict=True)
    }
    if actual_hashes != expected_hashes:
        raise ValueError("TextWorld task content differs from the frozen manifest")
    train = tuple(
        str(record["task_id"]) for record in records if record.get("split") == "train"
    )
    evaluation = tuple(
        str(record["task_id"]) for record in records if record.get("split") == "eval"
    )
    if not train or not evaluation or len(train) + len(evaluation) != len(records):
        raise ValueError("TextWorld manifest must contain non-empty train and eval splits")
    return FrozenTextWorldDataset(
        root=dataset_root,
        train_task_ids=train,
        evaluation_task_ids=evaluation,
        directory_by_task_id=dict(zip(task_ids, directories, strict=True)),
        content_sha256=actual_hashes,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-tasks", type=int, default=256)
    parser.add_argument("--evaluation-tasks", type=int, default=32)
    parser.add_argument(
        "--tw-make",
        default=None,
        help=(
            "Existing tw-make executable; omit to bootstrap the pinned "
            "generator environment"
        ),
    )
    parser.add_argument(
        "--generator-venv",
        type=Path,
        default=Path.home() / ".cache" / "fireworks-textworld-generator",
        help="Pinned generator venv created when --tw-make is omitted",
    )
    cooking = parser.add_argument_group("cooking")
    cooking.add_argument(
        "--recipe", type=int, default=CALIBRATED_COOKING_SETTINGS["recipe"]
    )
    cooking.add_argument(
        "--take", type=int, default=CALIBRATED_COOKING_SETTINGS["take"]
    )
    cooking.add_argument(
        "--go",
        type=int,
        choices=(1, 6, 9, 12),
        default=CALIBRATED_COOKING_SETTINGS["go"],
    )
    cooking.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=CALIBRATED_COOKING_SETTINGS["open"],
        dest="open_",
    )
    cooking.add_argument(
        "--cook",
        action=argparse.BooleanOptionalAction,
        default=CALIBRATED_COOKING_SETTINGS["cook"],
    )
    cooking.add_argument(
        "--cut",
        action=argparse.BooleanOptionalAction,
        default=CALIBRATED_COOKING_SETTINGS["cut"],
    )
    cooking.add_argument(
        "--drop",
        action=argparse.BooleanOptionalAction,
        default=CALIBRATED_COOKING_SETTINGS["drop"],
    )
    cooking.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default=CALIBRATED_COOKING_SETTINGS["split"],
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    settings = {
        "recipe": args.recipe,
        "take": args.take,
        "go": args.go,
        "open": args.open_,
        "cook": args.cook,
        "cut": args.cut,
        "drop": args.drop,
        "split": args.split,
    }
    tw_make = (
        args.tw_make
        if args.tw_make is not None
        else str(ensure_generator_environment(args.generator_venv))
    )
    dataset = generate_dataset(
        args.output,
        seed=args.seed,
        train_tasks=args.train_tasks,
        evaluation_tasks=args.evaluation_tasks,
        settings=settings,
        tw_make=tw_make,
    )
    print(
        f"generated {len(dataset.train_task_ids)} train and "
        f"{len(dataset.evaluation_task_ids)} evaluation cooking tasks "
        f"at {dataset.root}"
    )


if __name__ == "__main__":
    main()
