#!/usr/bin/env node
// Advisor — a critique-only second opinion from a frontier reviewer (Claude),
// callable by any coding agent. The agent does the work; the advisor reviews the
// working git diff before it finishes. It cannot edit files.
//
//   node advisor.mjs review --question "..." [--files a.ts,b.ts]
//
// Reviewer config (separate from your Fireworks worker — see README.md):
//   ADVISOR_API_KEY    Anthropic key, sk-ant-...   [required]
//   ADVISOR_BASE_URL   endpoint  [default: https://api.anthropic.com]
//   ADVISOR_MODEL      model     [default: claude-opus-4-8]

import { readFileSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import process from "node:process";

const API_KEY = process.env.ADVISOR_API_KEY;
const BASE_URL = (process.env.ADVISOR_BASE_URL || "https://api.anthropic.com").replace(/\/+$/, "");
const MODEL = process.env.ADVISOR_MODEL || "claude-opus-4-8";

// The system prompt IS the design: a skeptic that audits the diff as ground
// truth, not the agent's prose. (From the benchmarked harness — see the report.)
const SYSTEM_PROMPT =
  "You are a critique-only senior engineering reviewer. You see the calling " +
  "agent's question and current work (source files and the working diff). You " +
  "CANNOT modify files; you only return advice. " +
  "Independence: do NOT accept the agent's framing, arithmetic, or boundaries " +
  "(date windows, counts, inclusive/exclusive ranges) - recompute them yourself " +
  "from the raw data; the agent's question may embed its own bug. " +
  "Verifiability: if you cannot verify a claim because you lack the original or " +
  "reference material, say CANNOT VERIFY explicitly - never assert correctness " +
  "you could not check. " +
  "Confidence: score each issue 0-100 (0=false positive; 25=could not verify; " +
  "50=real but minor; 75=verified and important; 100=certain). " +
  "Be specific: name the file, function, or line. " +
  "Structure the response in four sections, ~300 words total: " +
  "(1) Critical issues - ONLY issues scoring >=80, each with its score. " +
  "(2) Concrete suggested fixes for those. " +
  "(3) Low-confidence notes (no action) - everything below 80. " +
  "(4) Verification: what you could and could not verify. " +
  "EVIDENCE: the <worktree> section (git status + diff) is ground truth for the " +
  "current uncommitted state - the agent's prose claims of edits are not. Treat " +
  "any claim that tests/build pass as CANNOT VERIFY unless verbatim output from a " +
  "run after the last edit is in the context. A claim contradicted by output you " +
  "can see is itself a critical issue.";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith("--")) continue;
    out[argv[i].slice(2)] = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
  }
  return out;
}

// Ground truth: the working git diff at cwd. The diff is what this tool is built
// around, so if it can't be read — missing git, not a repo, or a diff larger than
// execFileSync's 1MB default — warn loudly instead of silently dropping it.
function worktree() {
  const git = (a) => execFileSync("git", a, { encoding: "utf8", timeout: 15000, maxBuffer: 64 * 1024 * 1024 });
  try {
    return `## git status\n${git(["status", "--porcelain"]).trimEnd() || "(clean)"}\n\n` +
           `## git diff HEAD\n${git(["diff", "HEAD", "--no-color"]).trimEnd() || "(none)"}`;
  } catch (e) {
    console.error(`advisor: WARNING — no git diff attached (${e.message.split("\n")[0].slice(0, 140)}); reviewing without it.`);
    return "";
  }
}

function sources(arg) {
  if (!arg || arg === "true") return "";
  return arg.split(",").map((p) => p.trim()).filter(Boolean).map((p) =>
    existsSync(p) ? `## ${p}\n\n${readFileSync(p, "utf8")}` : `## ${p}\n\n(not found)`
  ).join("\n\n");
}

async function review({ question, files }) {
  const sections = {
    question: question || "(no specific question - give an overall critical review)",
    sources: sources(files),
    worktree: worktree(),
  };
  const userMessage = Object.entries(sections)
    .filter(([, v]) => v && v.trim())
    .map(([k, v]) => `<${k}>\n${v}\n</${k}>`)
    .join("\n");

  const body = { model: MODEL, max_tokens: 8192, system: SYSTEM_PROMPT, messages: [{ role: "user", content: userMessage }] };
  if (/^claude/i.test(MODEL)) body.thinking = { type: "adaptive" }; // Anthropic-only field

  const res = await fetch(`${BASE_URL}/v1/messages`, {
    method: "POST",
    headers: { "x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`advisor ${res.status}: ${(await res.text()).slice(0, 400)}`);
  const data = await res.json();
  return (data.content ?? []).map((b) => b.text || "").join("").trim() || "(advisor returned no text)";
}

if (!API_KEY) {
  console.error("advisor not configured - set ADVISOR_API_KEY (your Anthropic key for the Claude reviewer). See README.md.");
  process.exit(2);
}
if (process.argv[2] !== "review") {
  console.error('usage: node advisor.mjs review --question "..." [--files a,b]');
  process.exit(64);
}
try {
  console.log(await review(parseArgs(process.argv.slice(3))));
} catch (err) {
  console.error(`advisor failed: ${API_KEY ? err.message.replaceAll(API_KEY, "[REDACTED]") : err.message}`);
  process.exit(1);
}
