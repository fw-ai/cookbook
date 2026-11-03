| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 3.3% | 2.9% | 2.9% | 5 |
| base | 0.0% | 0.0% | 0.0% | 1 |
| adapter+prompt | 80.0% | 77.1% | 89.6% | 5 |
| adapter | 78.3% | 74.2% | 88.8% | 8 |
| base+mask | 18.3% | 22.1% | 87.9% | 4 |

Joint success, adapter without instruction minus prompted base: +71.2% (95% paired bootstrap interval +62.5% to +79.2%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
