Metric defs: greedy = 1-shot deterministic; best_verified@k = sample k, output the verifier-confirmed-correct one (deployable, exact verifier); maj@k = majority-vote answer.

| Model | train(in-dist) greedy | hard-test greedy | bv@1 | bv@4 | bv@8 | bv@16 | maj@16 | halluc↓ | Countdown OOD |
|---|---|---|---|---|---|---|---|---|---|
| divB20 | 32% | 0% | 2% | 6% | 13% | 16% | 1% | nan% | nan% |
