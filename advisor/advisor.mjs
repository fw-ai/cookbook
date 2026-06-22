#!/usr/bin/env node
// Fireworks Advisor — a critique-only "second opinion" from a stronger reviewer
// model, callable by ANY coding agent. The agent does the work; the advisor
// reviews the working git diff before the agent finishes. It cannot edit files.
//
// Run it from inside your project repo (your agent runs this, or you do):
//   node advisor.mjs review --question "..." [--focus "..."] [--files a.ts,b.ts]
//
// Configure the reviewer (defaults to Fireworks serverless inference):
//   ADVISOR_API_KEY    Fireworks API key, fw_...                    [required]
//   ADVISOR_BASE_URL   Anthropic-compatible endpoint   [default: Fireworks serverless]
//   ADVISOR_MODEL      reviewer model                  [default: a strong Fireworks model]
//
// To use a frontier reviewer instead (the report's headline setup), point
// ADVISOR_BASE_URL at https://api.anthropic.com with an Anthropic key and
// ADVISOR_MODEL=claude-opus-4-8. See README.md.

import { readFileSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import process from "node:process";

const API_KEY = process.env.ADVISOR_API_KEY;
const BASE_URL = (process.env.ADVISOR_BASE_URL || "https://api.fireworks.ai/inference").replace(/\/+$/, "");
const MODEL = process.env.ADVISOR_MODEL || "accounts/fireworks/models/glm-5p2";

// ---------------------------------------------------------------------------
// The system prompt IS the design: a skeptic, not a cheerleader. It distrusts
// the agent's prose and audits the diff as ground truth. (Preserved verbatim
// from the benchmarked harness — see the report.)
// ---------------------------------------------------------------------------
const EVIDENCE_RULES =
  " EVIDENCE RULES: the <worktree> section (git status + recent log + " +
  "diff), when present, is ground truth for the current UNCOMMITTED state " +
  "of the repo - the agent's prose claims of edits are not. A change " +
  "absent from the diff may have been committed (check the log) or may " +
  "predate the session. " +
  "Treat any claim that tests/build/verification pass as CANNOT VERIFY " +
  "unless the verbatim command output from a run made after the last edit " +
  "appears in the context; then demand that run. A claim contradicted by " +
  "output you can see is itself a critical issue. If the context contains " +
  "an earlier advisor critique, first audit each of its critical issues " +
  "against the worktree (diff and log) and mark it APPLIED or NOT APPLIED; " +
  "NOT APPLIED criticals remain critical.";

const SYSTEM_PROMPT =
  "You are a critique-only senior engineering reviewer. You see the calling " +
  "agent's question, optional focus area, and current work (source files and " +
  "the working diff). You CANNOT modify files; you only return advice. " +
  "Independence: do NOT accept the agent's framing, arithmetic, or boundaries " +
  "(date windows, counts, inclusive/exclusive ranges) - recompute them " +
  "yourself from the raw data in the context; the agent's question may embed " +
  "its own bug. " +
  "Verifiability: if you cannot verify a claim because you lack the original " +
  "or reference material, say CANNOT VERIFY explicitly - never assert " +
  "correctness you could not check. " +
  "Confidence: score each issue 0-100 (0=false positive; 25=could not " +
  "verify; 50=real but minor; 75=verified and important; 100=certain). " +
  "Be specific: name the file, function, or line when calling out issues. " +
  "Structure the response in four sections, ~300 words total: " +
  "(1) Critical issues - ONLY issues scoring >=80, each with its score. " +
  "(2) Concrete suggested fixes for those critical issues. " +
  "(3) Low-confidence notes (no action required) - everything below 80. " +
  "(4) Verification: one line each on what you could and could not verify." +
  EVIDENCE_RULES;

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith("--")) continue;
    const k = argv[i].slice(2);
    out[k] = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
  }
  return out;
}

// Ground truth: the working git diff at the agent's cwd. Harness-agnostic —
// every coding agent produces the same diff. Returns "" outside a git repo.
function worktree() {
  const git = (a) => execFileSync("git", a, { encoding: "utf8", timeout: 15000 });
  try {
    const status = git(["status", "--porcelain"]).trimEnd();
    const log = git(["log", "--oneline", "-10"]).trimEnd();
    const diff = git(["diff", "HEAD", "--no-color"]).trimEnd();
    return `## git status --porcelain\n${status || "(clean)"}\n\n` +
           `## git log --oneline -10\n${log}\n\n` +
           `## git diff HEAD\n${diff || "(no uncommitted tracked changes)"}`;
  } catch {
    return "";
  }
}

function sources(filesArg) {
  if (!filesArg || filesArg === "true") return "";
  return filesArg.split(",").map((p) => p.trim()).filter(Boolean).map((p) =>
    existsSync(p) ? `## ${p}\n\n${readFileSync(p, "utf8")}` : `## ${p}\n\n(file not found)`
  ).join("\n\n---\n\n");
}

async function review({ question, focus, files, context }) {
  const sections = {
    question: question || "(no specific question - give an overall critical review)",
    focus: focus || "(no specific focus)",
    context: context || "",
    sources: sources(files),
    worktree: worktree(),
  };
  const userMessage = Object.entries(sections)
    .filter(([, v]) => v && v.trim())
    .map(([k, v]) => `<${k}>\n${v}\n</${k}>`)
    .join("\n");

  const body = {
    model: MODEL,
    max_tokens: 8192,
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userMessage }],
  };
  // Adaptive thinking is an Anthropic-only field; only send it to Claude models.
  if (/^claude/i.test(MODEL)) body.thinking = { type: "adaptive" };

  const res = await fetch(`${BASE_URL}/v1/messages`, {
    method: "POST",
    headers: { "x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`advisor ${res.status}: ${(await res.text()).slice(0, 400)}`);
  const data = await res.json();
  return (data.content ?? []).map((b) => b.text || "").join("").trim() || "(advisor returned no text)";
}

async function main() {
  if (!API_KEY) {
    console.error("advisor not configured - set ADVISOR_API_KEY (your Fireworks key). See README.md.");
    process.exit(2);
  }
  const args = parseArgs(process.argv.slice(3)); // argv[2] is the subcommand "review"
  if (process.argv[2] !== "review") {
    console.error('usage: node advisor.mjs review --question "..." [--focus "..."] [--files a,b] [--context "..."]');
    process.exit(64);
  }
  console.log(await review(args));
}

main().catch((err) => {
  const msg = API_KEY ? err.message.replaceAll(API_KEY, "[REDACTED]") : err.message;
  console.error(`advisor failed: ${msg}`);
  process.exit(1);
});
