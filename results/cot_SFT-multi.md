Generative CoT self-critic (k=16, T=1.0); test=data/game24_official/test.parquet
selfcritic_cot@k = model reasons step-by-step, picks first candidate it marks CORRECT (no oracle).
verifier_* = how good the model is at JUDGING a candidate when allowed to compute.

| Model | greedy | selfcritic_cot@16 | best_verified@16 | maj@16 | verif_prec | verif_rec | verif_acc |
|---|---|---|---|---|---|---|---|
| SFT-multi | 8.00% | 6.00% | 50.00% | 5.00% | 5.95% | 98.86% | 7.90% |
