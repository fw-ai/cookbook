import { randomUUID } from "node:crypto";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const toolTimeoutSeconds = Number.parseInt(process.env.FIREWORKS_TITO_TOOL_TIMEOUT_SECONDS ?? "900", 10);
if (!Number.isFinite(toolTimeoutSeconds) || toolTimeoutSeconds < 1) {
	throw new Error("FIREWORKS_TITO_TOOL_TIMEOUT_SECONDS must be a positive integer");
}
const toolProfile = process.env.FIREWORKS_TITO_TOOL_PROFILE ?? "coding";
if (toolProfile !== "coding" && toolProfile !== "textworld") {
	throw new Error(`unsupported FIREWORKS_TITO_TOOL_PROFILE: ${toolProfile}`);
}

const blockedTextWorldCommands = new Set(["quit", "restart", "restore", "save", "script", "transcript", "undo"]);

function validateTextWorldAction(action: string): string {
	const normalized = action.trim();
	if (!normalized || normalized.length > 256) {
		throw new Error("TextWorld action must contain 1-256 characters");
	}
	if (!/^[A-Za-z0-9][A-Za-z0-9 .,'-]*$/.test(normalized)) {
		throw new Error("TextWorld action contains unsupported characters");
	}
	const verb = normalized.split(/\s+/, 1)[0]?.toLowerCase() ?? "";
	if (blockedTextWorldCommands.has(verb)) {
		throw new Error(`TextWorld meta-command is not allowed: ${verb}`);
	}
	return normalized;
}

// Pi invokes before_provider_headers once for a logical provider request and
// reuses the resulting headers for provider-level HTTP retries. Higher-level
// assistant retry is disabled by the cookbook-generated settings.json.
export default function (pi: ExtensionAPI) {
	let compactionDepth = 0;

	if (toolProfile === "textworld") {
		pi.registerTool({
			name: "textworld_action",
			label: "TextWorld action",
			description:
				"Perform exactly one natural-language action in the current TextWorld game and return the observation.",
			promptSnippet: "Perform one action in the current TextWorld game",
			promptGuidelines: [
				"Use textworld_action for every game action.",
				"Use textworld_reset only when restarting the current game is necessary.",
			],
			parameters: Type.Object({
				action: Type.String({
					minLength: 1,
					maxLength: 256,
					description: 'One game action, for example "go north" or "read cookbook".',
				}),
			}),
			executionMode: "sequential",
			async execute(_toolCallId, params, signal) {
				const action = validateTextWorldAction(params.action);
				const result = await pi.exec("textworld", [action], {
					signal,
					timeout: toolTimeoutSeconds * 1000,
				});
				const output = result.stdout || result.stderr || "(no observation)";
				return {
					content: [{ type: "text", text: output }],
					details: { code: result.code, killed: result.killed },
					isError: result.code !== 0 || result.killed,
				};
			},
		});
		pi.registerTool({
			name: "textworld_reset",
			label: "Reset TextWorld",
			description: "Reset the current TextWorld game to its initial state.",
			promptSnippet: "Reset the current TextWorld game",
			parameters: Type.Object({}),
			executionMode: "sequential",
			async execute(_toolCallId, _params, signal) {
				const result = await pi.exec("textworld-reset", [], {
					signal,
					timeout: toolTimeoutSeconds * 1000,
				});
				const output = result.stdout || result.stderr || "(game reset)";
				return {
					content: [{ type: "text", text: output }],
					details: { code: result.code, killed: result.killed },
					isError: result.code !== 0 || result.killed,
				};
			},
		});
		pi.on("session_start", () => {
			pi.setActiveTools(["textworld_action", "textworld_reset"]);
		});
	}

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
		if (!Object.prototype.hasOwnProperty.call(input, "timeout")) {
			input.timeout = toolTimeoutSeconds;
			return;
		}
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
