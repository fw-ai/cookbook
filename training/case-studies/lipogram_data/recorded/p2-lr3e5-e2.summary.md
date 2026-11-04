| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 1.7% | 7.1% | 7.1% | 5 |
| base | 0.0% | 0.0% | 0.0% | 5 |
| adapter+prompt | 40.0% | 34.2% | 37.5% | 10 |
| adapter | 30.0% | 28.3% | 34.6% | 5 |
| base+mask | 26.7% | 20.4% | 87.1% | 7 |

Joint success, adapter without instruction minus prompted base: +21.2% (95% paired bootstrap interval +14.2% to +29.2%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
