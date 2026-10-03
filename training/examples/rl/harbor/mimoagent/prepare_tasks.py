"""Copy Harbor tasks and bake the pinned mimoagent harness into their images.

mimoagent pins ``requires-python == "3.12.*"``, which task images do not
provide reliably, so the layer builds a standalone CPython 3.12 with uv and
installs the pinned upstream commit into its own venv at ``/opt/mimoagent``.
Nothing here redistributes mimoagent source; the pinned commit is fetched
from the upstream repository at image build time (see constants.py for
attribution).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from training.examples.rl.harbor.tito.prepare_tasks import (
    build_python_sidecar_suffix,
    prepare_with_installer,
)

from .constants import PINNED_MIMOAGENT_COMMIT

_MARKER = "# Added by fireworks TITO harbor.mimoagent.prepare_tasks"
_DRIVER_NAME = "mimoagent_driver.py"
_MCP_DIR_NAME = "mimoagent-mcp"
# uv goes into the sidecar venv (present, pip-enabled, no PEP 668 friction);
# the mimoagent venv it creates is fully standalone afterwards. The managed
# CPython and both venvs live under /opt and are world-readable: uv's default
# install dir is $HOME/.local, which a restored non-root USER cannot traverse.
# The pinned `mcp` SDK is what the task bundles' own mcp_bridge.py uses.
_MIMOAGENT_INSTALL = (
    # git is required for the pinned-commit install; not every task image has it.
    "RUN set -eu; \\\n"
    " if ! command -v git >/dev/null 2>&1; then"
    " if command -v apt-get >/dev/null 2>&1; then"
    " apt-get update && apt-get install -y --no-install-recommends git"
    " && rm -rf /var/lib/apt/lists/*;"
    " elif command -v dnf >/dev/null 2>&1; then dnf install -y git && dnf clean all;"
    " else echo 'no git and no known package manager' >&2; exit 1; fi; fi\n"
    "RUN set -eu; \\\n"
    " /opt/fireworks-tito/bin/pip install --no-cache-dir uv==0.12.17; \\\n"
    " UV_PYTHON_INSTALL_DIR=/opt/uv-python /opt/fireworks-tito/bin/uv python install 3.12; \\\n"
    " UV_PYTHON_INSTALL_DIR=/opt/uv-python /opt/fireworks-tito/bin/uv venv"
    " --python 3.12 /opt/mimoagent; \\\n"
    " UV_PYTHON_INSTALL_DIR=/opt/uv-python /opt/fireworks-tito/bin/uv pip install"
    " --python /opt/mimoagent/bin/python"
    f" 'git+https://github.com/XiaomiMiMo/mimoagent.git@{PINNED_MIMOAGENT_COMMIT}'"
    " 'mcp==1.30.0'; \\\n"
    " chmod -R a+rX /opt/uv-python /opt/mimoagent; \\\n"
    " /opt/mimoagent/bin/python -c"
    " 'import mimoagent, mimoagent.agents.bashonly, mimoagent.agents.cc,"
    " mimoagent.environments.local, mimoagent.models.openai_chat, mcp'; \\\n"
    " /opt/mimoagent/bin/python -c 'import mimoagent; print(mimoagent.__version__)'; \\\n"
    " if command -v runuser >/dev/null 2>&1; then"
    " runuser -u nobody -- env HOME=/tmp MIMOAGENT_GLOBAL_CONFIG_DIR=/tmp/mimoagent-check"
    " /opt/mimoagent/bin/python -c 'import mimoagent';"
    " elif command -v su >/dev/null 2>&1; then"
    " su -s /bin/sh nobody -c 'HOME=/tmp MIMOAGENT_GLOBAL_CONFIG_DIR=/tmp/mimoagent-check"
    ' /opt/mimoagent/bin/python -c "import mimoagent"\';'
    " else echo 'no non-root user tool; skipping non-root import check'; fi\n"
    f"COPY {_DRIVER_NAME} /opt/mimoagent-driver.py\n"
    f"COPY {_MCP_DIR_NAME} /opt/{_MCP_DIR_NAME}\n"
)

_MCP_CLIENT_SOURCE = (
    Path(__file__).resolve().parents[1] / "datasets" / "mimo" / "mcp" / "client.py"
)


def _suffix(restore_user: str | None) -> str:
    suffix = build_python_sidecar_suffix(marker=_MARKER, restore_user=None)
    suffix += _MIMOAGENT_INSTALL
    if restore_user is not None:
        suffix += f"USER {restore_user}\n"
    return suffix


def prepare(
    source: Path,
    destination: Path,
    *,
    base_image: str | None = None,
    task_names: list[str] | None = None,
) -> list[Path]:
    prepared = prepare_with_installer(
        source,
        destination,
        marker=_MARKER,
        suffix_builder=_suffix,
        base_image=base_image,
        task_names=task_names,
    )
    driver = Path(__file__).with_name("driver.py")
    tools = Path(__file__).with_name("mcp_tools.py")
    for task in prepared:
        shutil.copy(driver, task / "environment" / _DRIVER_NAME)
        mcp_dir = task / "environment" / _MCP_DIR_NAME
        mcp_dir.mkdir(exist_ok=True)
        # Flat names in the image: no package layout, no shadowing of the SDK.
        shutil.copy(_MCP_CLIENT_SOURCE, mcp_dir / "mcp_client.py")
        shutil.copy(tools, mcp_dir / "mcp_tools.py")
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--base-image",
        default=None,
        help="Optional immutable image@sha256 base for a single prepared task",
    )
    args = parser.parse_args()
    prepared = prepare(args.source, args.destination, base_image=args.base_image)
    print(
        f"prepared {len(prepared)} Harbor task image contexts in "
        f"{args.destination} with the pinned mimoagent harness"
    )


if __name__ == "__main__":
    main()
