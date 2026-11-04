| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 1.7% | 2.9% | 3.8% | 5 |
| base | 0.0% | 0.0% | 0.0% | 2 |
| adapter+prompt | 86.7% | 82.1% | 92.9% | 5 |
| adapter | 86.7% | 83.3% | 94.2% | 7 |
| base+mask | 26.7% | 20.4% | 88.3% | 7 |

Joint success, adapter without instruction minus prompted base: +80.4% (95% paired bootstrap interval +73.3% to +86.7%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
