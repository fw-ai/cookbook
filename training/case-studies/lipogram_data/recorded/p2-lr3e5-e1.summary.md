| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 1.7% | 4.6% | 5.4% | 2 |
| base | 0.0% | 0.0% | 0.0% | 0 |
| adapter+prompt | 35.0% | 29.2% | 33.3% | 7 |
| adapter | 26.7% | 26.2% | 33.3% | 8 |
| base+mask | 33.3% | 23.3% | 90.4% | 6 |

Joint success, adapter without instruction minus prompted base: +21.7% (95% paired bootstrap interval +13.8% to +29.6%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
