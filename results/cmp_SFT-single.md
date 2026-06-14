Verify-by-compute self-critic (few-shot + forced computation, k=16); test=data/game24_official/test.parquet

| Model | greedy | selfcritic_cmp@16 | best_verified@16 | maj@16 | verif_prec | verif_rec | verif_acc |
|---|---|---|---|---|---|---|---|
| SFT-single | 8.00% | 40.00% | 47.00% | 12.00% | 86.87% | 81.90% | 97.99% |
