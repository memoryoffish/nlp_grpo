Verify-by-compute self-critic (few-shot + forced computation, k=16); test=data/game24_official/test.parquet

| Model | greedy | selfcritic_cmp@16 | best_verified@16 | maj@16 | verif_prec | verif_rec | verif_acc |
|---|---|---|---|---|---|---|---|
| SFT-multi | 8.00% | 31.00% | 52.00% | 7.00% | 57.78% | 50.98% | 94.50% |
