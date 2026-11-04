"""Fresh-import checks run with both the current SDK and the minimum wheel."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("renderer_first", [False, True])
def test_loader_imports_have_no_cycle_or_new_sdk_requirement(renderer_first: bool) -> None:
    modules = [
        "training.utils",
        "training.utils.supervised",
        "training.utils.tokenizers",
        "training.renderer._tokenizer_utils",
        "training.renderer._image_processing_utils",
        "training.renderer.verifier.utils.tokenizer",
        "training.renderer.verifier.utils.hf_parity",
        "training.examples.tools.verify_logprobs",
    ]
    if renderer_first:
        modules = modules[3:] + modules[:3]
    code = (
        "import importlib\n"
        f"for name in {modules!r}: importlib.import_module(name)\n"
        "import training.utils.tokenizers as loaders\n"
        "import training.tokenizer_identity as identity\n"
        "assert loaders.requires_code_free_tokenizer_loading.__module__ == 'training.tokenizer_identity'\n"
        "assert identity.is_deepseek_v4_tokenizer_source(None) is False\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")
