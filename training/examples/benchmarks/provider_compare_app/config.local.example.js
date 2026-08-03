// Optional key injection for standalone.html (no backend).
//
// Copy to `config.local.js` (gitignored) and fill in your keys, OR just paste
// keys into the app's Settings panel instead (stored in localStorage).
//
// Generate this file straight from training/.env:
//   cd training/examples/benchmarks/provider_compare_app
//   awk -F= 'BEGIN{print "window.PROVIDER_KEYS = {"} \
//     /^(FIREWORKS|BASETEN|TOGETHER)_API_KEY=/{printf "  %s: \"%s\",\n",$1,$2} \
//     END{print "};"}' ../../../.env > config.local.js
//
// Keys here are loaded into the browser and sent directly to each provider.
// Local/personal use only — never deploy this file to a public host.
window.PROVIDER_KEYS = {
  FIREWORKS_API_KEY: "fw-...",
  BASETEN_API_KEY: "...",
  TOGETHER_API_KEY: "...",
};
