"""Verify that the built cookbook artifact imports with published dependencies.

Run this script from outside the checkout after ``pip install .``. It discovers
the base modules shipped from ``training.recipes`` and ``training.utils`` in the
checkout, then imports them from the installed wheel. No credentials, network,
or GPU are required.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version
from pathlib import Path

import training

CHECKOUT_PACKAGE = Path(__file__).resolve().parents[1]
INSTALLED_PACKAGE = Path(training.__file__).resolve().parent

# These modules intentionally require optional dependency groups (``eval``,
# ``sdft``) that are not installed by the base ``pip install .`` smoke.
OPTIONAL_MODULES = {
    "training.utils.rl.rollout.eval_protocol",
    "training.recipes.sdft_loop",
}


def _checkout_modules(package_dir: Path) -> list[str]:
    modules = []
    for path in package_dir.rglob("*.py"):
        relative = path.relative_to(CHECKOUT_PACKAGE).with_suffix("")
        parts = relative.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(("training", *parts)))
    return sorted(set(modules) - OPTIONAL_MODULES)


def main() -> None:
    if INSTALLED_PACKAGE == CHECKOUT_PACKAGE:
        raise RuntimeError(
            "Imported training from the checkout instead of the installed artifact"
        )

    modules = []
    for package_name in ("recipes", "utils"):
        modules.extend(_checkout_modules(CHECKOUT_PACKAGE / package_name))

    for module_name in modules:
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        if module_path and Path(module_path).resolve().is_relative_to(CHECKOUT_PACKAGE):
            raise RuntimeError(
                f"Imported {module_name} from the checkout: {module_path}"
            )

    bundled_files = [
        "examples/sft/text2sql_dataset.jsonl",
        "examples/sft/food_reasoning.jsonl",
        "examples/sft/data/support_triage_train.jsonl",
        "examples/rl/deepmath/dataset.jsonl",
        "examples/rl/frozen_lake/seeds.jsonl",
        "examples/embedding/retrieval_pairs.jsonl",
        "examples/harbor_rl/tasks/mimo/music_scorer/baselines/ref_full4k.json",
        "examples/distillation/serverless_k3_privileged/data/gsm8k_privileged.jsonl",
        "examples/distillation/serverless_k3_privileged/data/gsm8k_200.jsonl",
        "examples/archive/countdown/data/countdown_train.jsonl",
        "examples/archive/eval_protocol_chat/train.jsonl",
        "tests/_encoding_dsv41_oracle.LICENSE",
    ]
    # Rendering names these assets directly. A glob would silently omit an
    # asset missing from the checkout and let an incomplete wheel pass.
    bundled_files.extend(
        f"examples/rl/frozen_lake/assets/img/{name}.png"
        for name in (
            "ice", "hole", "cracked_hole", "goal", "stool",
            "elf_left", "elf_down", "elf_right", "elf_up",
        )
    )
    for relative in bundled_files:
        source = CHECKOUT_PACKAGE / relative
        if not source.is_file():
            raise RuntimeError(f"Bundled example resource missing from checkout: {relative}")
        installed = INSTALLED_PACKAGE / relative
        if not installed.is_file() or installed.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Bundled example resource missing or changed: {relative}")

    print(f"verified {len(bundled_files)} bundled example resources")
    print(f"cookbook: {version('fireworks-training-cookbook')} ({INSTALLED_PACKAGE})")
    print(f"fireworks-ai: {version('fireworks-ai')}")
    print(f"imported {len(modules)} installed cookbook modules")


if __name__ == "__main__":
    main()
