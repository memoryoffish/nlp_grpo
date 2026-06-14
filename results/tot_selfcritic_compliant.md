ToT-aligned self-critic best-of-N (k=16, T=1.0); test=data/game24_official/test.parquet
selfcritic@k = model picks among k (no oracle, ToT); best_verified@k = oracle upper bound; maj@k = majority vote.

| Model | greedy | selfcritic@16 | best_verified@16 | maj@16 |
|---|---|---|---|---|
| SFT-multi | 8.00% | 6.00% | 51.00% | 8.00% |
| C-multi-n4-lr1e6 | 11.00% | 8.00% | 42.00% | 8.00% |
| A1-n8 | 12.00% | 12.00% | 34.00% | 11.00% |
| R1-puregrpo | 0.00% | 2.00% | 8.00% | 0.00% |
