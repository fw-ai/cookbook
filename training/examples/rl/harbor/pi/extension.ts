import { randomUUID } from "node:crypto";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const toolTimeoutSeconds = Number.parseInt(process.env.FIREWORKS_TITO_TOOL_TIMEOUT_SECONDS ?? "900", 10);
if (!Number.isFinite(toolTimeoutSeconds) || toolTimeoutSeconds < 1) {
	throw new Error("FIREWORKS_TITO_TOOL_TIMEOUT_SECONDS must be a positive integer");
}

// Pi invokes before_provider_headers once for a logical provider request and
// reuses the resulting headers for provider-level HTTP retries. Higher-level
// assistant retry is disabled by the cookbook-generated settings.json.
export default function (pi: ExtensionAPI) {
	let compactionDepth = 0;

	pi.on("session_before_compact", () => {
		compactionDepth += 1;
	});
	pi.on("session_compact", () => {
		compactionDepth = Math.max(0, compactionDepth - 1);
	});
	pi.on("session_compact_failed", () => {
		compactionDepth = Math.max(0, compactionDepth - 1);
	});
	pi.on("session_before_tree", () => {
		// V1 has no ancestry. This adapter rejects in-process tree navigation;
		// another policy loop must receive a fresh independent trajectory.
		return { cancel: true };
	});
	pi.on("session_before_fork", () => {
		// The shipped rollout does not load Pi's child/subagent extensions and
		// never shares this policy loop's trajectory with another process.
		return { cancel: true };
	});
	pi.on("tool_call", (event) => {
		if (event.toolName !== "bash") return;
		const input = event.input as { timeout?: number };
		if (!Object.prototype.hasOwnProperty.call(input, "timeout")) return;
		const requested = input.timeout;
		if (
			typeof requested !== "number" ||
			!Number.isFinite(requested) ||
			requested < 1 ||
			requested > toolTimeoutSeconds
		) {
			// Pi's pinned bash backend kills the detached process group on timeout.
			input.timeout = toolTimeoutSeconds;
		}
	});

	pi.on("before_provider_headers", (event) => {
		event.headers["Idempotency-Key"] = randomUUID();
	});

	pi.on("before_provider_request", (event) => {
		if (typeof event.payload !== "object" || event.payload === null || Array.isArray(event.payload)) {
			throw new Error("Pi produced a non-object provider payload");
		}
		return {
			...event.payload,
			_tito: {
				call_kind: compactionDepth > 0 ? "auxiliary" : "policy",
				classifier_source: compactionDepth > 0 ? "pi_compaction_hook" : "pi_policy_hook",
			},
		};
	});
}
