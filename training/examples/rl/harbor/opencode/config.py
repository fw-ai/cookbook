"""Harbor-independent configuration used by the OpenCode adapter."""

import hashlib
from pathlib import Path

SHELL_FIX_VERSION = "1.18.8-fw-shell.2"
SHELL_FIX_SHA256 = "6b160847e94b9ecfa608436d23497de08ff5c93f30d9174bcf3fb0c45134ca88"


def validate_shell_fix_binary(path: str | Path) -> Path:
    """Accept only the exact E2B-tested native shell-fix build, without executing it."""
    binary = Path(path).expanduser().resolve(strict=True)
    with binary.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != SHELL_FIX_SHA256:
        raise ValueError(
            f"OpenCode shell-fix binary checksum mismatch: {digest}; "
            f"expected {SHELL_FIX_SHA256} ({SHELL_FIX_VERSION})"
        )
    return binary

_TOOL_TIMEOUT_PLUGIN = r"""
export const FireworksTitoToolTimeout = async () => {
  const maximum = Number.parseInt(
    process.env.FIREWORKS_TITO_TOOL_TIMEOUT_MS ?? "",
    10,
  )
  if (!Number.isFinite(maximum) || maximum < 1) {
    throw new Error("FIREWORKS_TITO_TOOL_TIMEOUT_MS must be a positive integer")
  }
  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool !== "bash") return
      if (
        output.args === null ||
        typeof output.args !== "object" ||
        Array.isArray(output.args)
      ) return
      // OpenCode's default-timeout environment variable handles an omitted
      // argument without rewriting the model-authored tool call in history.
      if (!Object.prototype.hasOwnProperty.call(output.args, "timeout")) return
      const requested = Number(output.args.timeout)
      if (!Number.isFinite(requested) || requested < 1 || requested > maximum) {
        // OpenCode 1.18.8 interprets the bash timeout in milliseconds. Mutating
        // the normalized args object is required by that pinned hook contract.
        output.args.timeout = maximum
      }
    },
  }
}
"""

__all__ = ["_TOOL_TIMEOUT_PLUGIN", "SHELL_FIX_VERSION", "SHELL_FIX_SHA256", "validate_shell_fix_binary"]
