| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 0.0% | 2.5% | 2.5% | 3 |
| base | 0.0% | 0.0% | 0.0% | 4 |
| adapter+prompt | 35.0% | 32.5% | 37.9% | 4 |
| adapter | 26.7% | 27.9% | 33.8% | 9 |
| base+mask | 18.3% | 25.4% | 90.8% | 5 |

Joint success, adapter without instruction minus prompted base: +25.4% (95% paired bootstrap interval +17.5% to +33.8%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
