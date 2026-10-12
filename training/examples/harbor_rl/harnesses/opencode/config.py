"""Harbor-independent configuration used by the OpenCode adapter."""

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

__all__ = ["_TOOL_TIMEOUT_PLUGIN"]
