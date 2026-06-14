Metric defs: greedy = 1-shot deterministic; best_verified@k = sample k, output the verifier-confirmed-correct one (deployable, exact verifier); maj@k = majority-vote answer.

| Model | train(in-dist) greedy | hard-test greedy | bv@1 | bv@2 | bv@4 | bv@8 | bv@16 | maj@16 | halluc↓ | Countdown OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| A4c-lr5e6 | 9% | 11% | 10% | 14% | 15% | 17% | 18% | 11% | 0% | 8% |
| R1-puregrpo | 4% | 0% | 1% | 3% | 3% | 8% | 13% | 1% | 0% | 4% |
