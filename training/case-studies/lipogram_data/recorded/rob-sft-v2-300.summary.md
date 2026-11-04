| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 5.0% | 4.6% | 4.6% | 5 |
| base | 0.0% | 0.0% | 0.0% | 2 |
| adapter+prompt | 6.7% | 7.1% | 7.9% | 0 |
| adapter | 0.0% | 0.8% | 0.8% | 3 |
| base+mask | 23.3% | 22.9% | 87.1% | 7 |

Joint success, adapter without instruction minus prompted base: -3.8% (95% paired bootstrap interval -7.1% to -0.8%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
