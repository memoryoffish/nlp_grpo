ToT-faithful self-critic (value_last_step_prompt, few-shot, n_eval=3, k=16); test=data/game24_official/test.parquet
selfcritic_tot@k = model self-judges via ToT few-shot value prompt, picks best (no oracle).

| Model | greedy | selfcritic_tot@16 | best_verified@16 | maj@16 | verif_prec | verif_rec | verif_acc |
|---|---|---|---|---|---|---|---|
| C | 11.00% | 11.00% | 42.00% | 10.00% | 11.08% | 97.71% | 13.95% |
