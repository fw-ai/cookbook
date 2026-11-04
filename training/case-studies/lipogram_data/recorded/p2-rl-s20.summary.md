| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 5.0% | 5.0% | 6.2% | 2 |
| base | 0.0% | 0.0% | 0.0% | 2 |
| adapter+prompt | 76.7% | 67.5% | 80.4% | 3 |
| adapter | 76.7% | 70.0% | 85.0% | 3 |
| base+mask | 28.3% | 20.8% | 89.2% | 4 |

Joint success, adapter without instruction minus prompted base: +65.0% (95% paired bootstrap interval +55.8% to +73.8%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
