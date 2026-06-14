Metric defs: greedy = 1-shot deterministic; best_verified@k = sample k, output the verifier-confirmed-correct one (deployable, exact verifier); maj@k = majority-vote answer.

| Model | train(in-dist) greedy | hard-test greedy | bv@1 | bv@2 | bv@4 | bv@8 | bv@16 | maj@16 | halluc↓ | Countdown OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| A3-shaped | 22% | 12% | 11% | 16% | 24% | 29% | 41% | 11% | 0% | 11% |
| A4a-lr5e7 | 19% | 12% | 15% | 18% | 23% | 37% | 56% | 12% | 0% | 4% |
| A4b-lr2e6 | 17% | 9% | 7% | 9% | 12% | 18% | 23% | 9% | 0% | 2% |
