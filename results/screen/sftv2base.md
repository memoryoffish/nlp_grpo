Metric defs: greedy = 1-shot deterministic; best_verified@k = sample k, output the verifier-confirmed-correct one (deployable, exact verifier); maj@k = majority-vote answer.

| Model | train(in-dist) greedy | hard-test greedy | bv@1 | bv@4 | bv@8 | bv@8 | maj@8 | halluc↓ | Countdown OOD |
|---|---|---|---|---|---|---|---|---|---|
| sftv2base | 27% | 0% | 1% | 5% | 7% | 7% | 2% | nan% | nan% |
