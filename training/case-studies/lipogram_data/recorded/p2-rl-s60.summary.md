| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 1.7% | 4.6% | 4.6% | 3 |
| base | 0.0% | 0.0% | 0.0% | 3 |
| adapter+prompt | 85.0% | 81.7% | 91.2% | 6 |
| adapter | 86.7% | 82.1% | 91.7% | 5 |
| base+mask | 21.7% | 20.4% | 86.7% | 8 |

Joint success, adapter without instruction minus prompted base: +77.5% (95% paired bootstrap interval +70.4% to +84.2%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
