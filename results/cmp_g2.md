Metric defs: greedy = 1-shot deterministic; best_verified@k = sample k, output the verifier-confirmed-correct one (deployable, exact verifier); maj@k = majority-vote answer.

| Model | train(in-dist) greedy | hard-test greedy | bv@1 | bv@2 | bv@4 | bv@8 | bv@16 | maj@16 | halluc↓ | Countdown OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| C-multi-n4-lr1e6 | 19% | 9% | 13% | 16% | 21% | 32% | 45% | 12% | 0% | 4% |
| A1-n8 | 19% | 13% | 14% | 18% | 20% | 25% | 31% | 10% | 0% | 4% |
| A2-single | 18% | 12% | 17% | 19% | 27% | 32% | 39% | 16% | 0% | 4% |
