Metric defs: greedy = 1-shot deterministic; best_verified@k = sample k, output the verifier-confirmed-correct one (deployable, exact verifier); maj@k = majority-vote answer.

| Model | train(in-dist) greedy | hard-test greedy | bv@1 | bv@2 | bv@4 | bv@8 | bv@16 | maj@16 | halluc↓ | Countdown OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 2% | 0% | 0% | 1% | 2% | 2% | 7% | 0% | 0% | 2% |
| SFT-multi | 22% | 7% | 6% | 11% | 17% | 28% | 42% | 6% | 0% | 3% |
| SFT-single | 20% | 9% | 5% | 9% | 21% | 29% | 45% | 4% | 0% | 9% |
