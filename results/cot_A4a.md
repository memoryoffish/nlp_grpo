Generative CoT self-critic (k=16, T=1.0); test=data/game24_official/test.parquet
selfcritic_cot@k = model reasons step-by-step, picks first candidate it marks CORRECT (no oracle).
verifier_* = how good the model is at JUDGING a candidate when allowed to compute.

| Model | greedy | selfcritic_cot@16 | best_verified@16 | maj@16 | verif_prec | verif_rec | verif_acc |
|---|---|---|---|---|---|---|---|
| A4a | 12.00% | 15.00% | 45.00% | 12.00% | 11.55% | 99.43% | 12.35% |
