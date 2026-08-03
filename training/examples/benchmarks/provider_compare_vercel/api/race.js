export const config = { runtime: "edge" };

const sse = (obj) => `data: ${JSON.stringify(obj)}\n\n`;

export default async function handler(req) {
  if (req.method !== "POST") return new Response("method not allowed", { status: 405 });

  // Optional shared-secret gate (set COMPARE_APP_SECRET in Vercel env).
  const secret = process.env.COMPARE_APP_SECRET || "";
  if (secret && req.headers.get("x-compare-secret") !== secret) {
    return new Response(JSON.stringify({ error: "unauthorized" }), { status: 401 });
  }

  let prompt = "", reasoning, maxTokens;
  try {
    const b = await req.json();
    prompt = (b.prompt || "").trim();
    reasoning = b.reasoning;          // "none" | "" | "low"|"medium"|"high"|"max" | undefined
    maxTokens = b.maxTokens;
  } catch {}
  if (!prompt) return new Response(JSON.stringify({ error: "prompt required" }), { status: 400 });

  let provs = [];
  try { provs = JSON.parse(process.env.PROVIDERS || "[]"); } catch {}
  if (!provs.length) {
    return new Response(JSON.stringify({ error: "PROVIDERS env var not configured" }), { status: 500 });
  }

  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const write = (o) => controller.enqueue(enc.encode(sse(o)));
      await Promise.all(provs.map((p, i) => raceOne(i, p, prompt, reasoning, maxTokens, write)));
      controller.enqueue(enc.encode("event: end\ndata: {}\n\n"));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "x-accel-buffering": "no",
    },
  });
}

async function raceOne(col, p, prompt, reasoning, maxTokens, write) {
  write({ column: col, type: "start", name: p.label || `Provider ${col + 1}` });
  const t0 = performance.now();
  let ttft = null, answerChars = 0, outTokens = null;
  try {
    const body = {
      model: p.model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens || p.max_tokens || 1024,
      stream: true,
      stream_options: { include_usage: true },
    };
    // Reasoning control. "none" merges the provider-specific disable payload
    // (Fireworks: reasoning_effort:none; Together: chat_template_kwargs.enable_thinking:false);
    // a concrete effort sets reasoning_effort; anything else leaves provider default.
    if (reasoning === "none") {
      Object.assign(body, p.reasoningOff || { reasoning_effort: "none" });
    } else if (reasoning) {
      body.reasoning_effort = reasoning;
    } else if (p.reasoning_effort) {
      body.reasoning_effort = p.reasoning_effort;
    }

    const r = await fetch(p.url, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${p.key}` },
      body: JSON.stringify(body),
    });
    if (!r.ok || !r.body) {
      const txt = await r.text().catch(() => "");
      write({ column: col, type: "error", error: `HTTP ${r.status}: ${txt.slice(0, 160)}` });
      write({ column: col, type: "done", tok_s: 0, out_tokens: 0 });
      return;
    }

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let ev; try { ev = JSON.parse(payload); } catch { continue; }
        const delta = ev.choices && ev.choices[0] ? ev.choices[0].delta : null;
        const piece = delta ? delta.content : null;
        // Reasoning field name varies: Fireworks=reasoning_content, Together=reasoning.
        const rPiece = delta ? (delta.reasoning_content ?? delta.reasoning) : null;
        if (piece || rPiece) {
          if (ttft === null) {
            ttft = (performance.now() - t0) / 1000;
            write({ column: col, type: "metric", ttft: +ttft.toFixed(3), tok_s: 0 });
          }
          if (rPiece) write({ column: col, type: "reason", text: rPiece });
          if (piece) {
            answerChars += piece.length;
            write({ column: col, type: "tok", text: piece });
            const ed = (performance.now() - t0) / 1000 - ttft;
            if (ed > 0) write({ column: col, type: "metric", ttft: +ttft.toFixed(3), tok_s: +((answerChars / 4) / ed).toFixed(1) });
          }
        }
        if (ev.usage && ev.usage.completion_tokens != null) outTokens = ev.usage.completion_tokens;
      }
    }
    const dt = (performance.now() - t0) / 1000;
    if (outTokens == null) outTokens = Math.round(answerChars / 4); // rough fallback
    const tps = (outTokens && ttft != null && dt > ttft) ? outTokens / (dt - ttft) : 0;
    write({ column: col, type: "done", ttft: ttft != null ? +ttft.toFixed(3) : null, tok_s: +tps.toFixed(1), out_tokens: outTokens, latency: +dt.toFixed(3) });
  } catch (e) {
    write({ column: col, type: "error", error: String((e && e.message) || e).slice(0, 200) });
    write({ column: col, type: "done", tok_s: 0, out_tokens: 0 });
  }
}
