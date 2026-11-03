# Arm comparison on 20 briefs, 2 draws, effort low

| arm | functional | policy claims/page | tokens (median) | patterns (clamp/grid/dark/svg) |
|---|---:|---:|---:|---|
| plain | 30/40 | 1.39 | 2209 | 0.69/0.46/0.03/1.0 |
| skill | 25/40 | 2.81 | 3414 | 1.0/0.65/0.0/1.0 |
| v1 | 28/40 | 1.73 | 2490 | 1.0/0.7/0.0/1.0 |
| v2a | 27/40 | 1.3 | 2368 | 1.0/0.55/0.0/1.0 |
| v2b | 31/40 | 0.85 | 2306 | 1.0/0.5/0.0/1.0 |

## Design preference (design judge, both orders)

| pair | first | second | tie |
|---|---:|---:|---:|
| plain vs v1 | 25 | 53 | 0 |
| plain vs v2b | 38 | 40 | 0 |
| plain vs v2a | 27 | 51 | 0 |
| plain vs skill | 11 | 67 | 0 |
| skill vs v1 | 60 | 20 | 0 |
| skill vs v2a | 62 | 18 | 0 |
| skill vs v2b | 69 | 11 | 0 |
| v1 vs v2a | 39 | 41 | 0 |
| v1 vs v2b | 51 | 29 | 0 |
| v2a vs v2b | 56 | 24 | 0 |

## Usefulness preference (original judge, overall)

| pair | first | second | tie |
|---|---:|---:|---:|
| plain vs v1 | 47 | 31 | 0 |
| plain vs v2b | 52 | 26 | 0 |
| plain vs v2a | 47 | 30 | 0 |
| plain vs skill | 31 | 47 | 0 |
| skill vs v1 | 57 | 21 | 0 |
| skill vs v2a | 50 | 29 | 0 |
| skill vs v2b | 57 | 22 | 0 |
| v1 vs v2a | 39 | 40 | 0 |
| v1 vs v2b | 50 | 30 | 0 |
| v2a vs v2b | 50 | 30 | 0 |

## Reading (2026-10-22)

Setup: identical 20 dev briefs, 2 draws per arm, effort low; plain/skill/v1 draws are the same files used in v1's
evaluation; v2a (168 rows, design bar) and v2b (134 rows, strict bar) trained with v1's settings. Design judge is the
art-direction judge calibrated to the owner's ordering; usefulness judge is the original brief-adherence judge.

- v2 did not close the design gap. skill > v1 60-20, skill > v2a 62-18, skill > v2b 69-11. v1 vs v2a is a tie (39-41);
  v2b regressed (v1 51-29 over it; v2b 40-38 vs plain).
- v2 improved factual fidelity: unsupported policy claims per page skill 2.81, v1 1.73, v2a 1.30, v2b 0.85; v2b has the
  best functional rate (31/40). Repair-and-select taught faithfulness.
- Pattern fingerprint: every adapter carries the skill's surface (clamp headlines 100%, no dark bands, grid 50-70%).
  The adapters copy the skill's layout; the judge's preference for the skill is driven by the per-page visual idea, which
  is different on every page and does not transfer from 150 examples of final pages.
- Selection audit was right that strong drafts were discarded (fact-check gate 15/18, missing-text gate 10/11), and the
  repair pass recovered 127 of them (88 needed no edit). Recovering them did not change the adapter's design score.

Recommendation: stop iterating on SFT selection; the gap is not which pages are picked. Honest framing for the gallery:
the adapter reproduces the skill's typography and structure with no prompt and better factual fidelity (v2b), and the
skill prompt remains the better designer. To pursue invention, the next experiment is revision training (weak draft ->
stronger rewrite of the same brief, teaching the move rather than the result), small, before DPO or RL.

Costs, from the ledger: steps 1-6 of this experiment about $11 (design judging $7, repairs $0.52, plain arm $0.50,
two training runs about $3). Study total to date $53.
