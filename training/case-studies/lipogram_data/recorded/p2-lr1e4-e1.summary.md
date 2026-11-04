| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |
|---|---:|---:|---:|---:|
| base+prompt | 8.3% | 2.1% | 2.1% | 3 |
| base | 0.0% | 0.0% | 0.0% | 5 |
| adapter+prompt | 40.0% | 36.2% | 40.4% | 7 |
| adapter | 35.0% | 29.2% | 36.2% | 6 |
| base+mask | 28.3% | 23.8% | 88.3% | 4 |

Joint success, adapter without instruction minus prompted base: +27.1% (95% paired bootstrap interval +18.3% to +36.2%).
Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.
