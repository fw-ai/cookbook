// Returns anonymized column labels to the browser. The real provider URLs,
// models, and keys live only in the PROVIDERS env var (server-side) and are
// never sent to the client.
export const config = { runtime: "edge" };

export default function handler() {
  let provs = [];
  try { provs = JSON.parse(process.env.PROVIDERS || "[]"); } catch {}
  const out = provs.map((p, i) => ({ id: i, name: p.label || `Provider ${i + 1}` }));
  return new Response(JSON.stringify(out), {
    headers: { "content-type": "application/json" },
  });
}
