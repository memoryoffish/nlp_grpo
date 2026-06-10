# Game24 ablation summary (consolidated)

## A. SFT v2 best-of-N ceiling on hard test (by temperature)

| temp | greedy | bv@8 | bv@16 | bv@32 | bv@64 |
|---|---|---|---|---|---|
| sftv2_T1.0 | 3% | 12% | 16% | 26% | 37% |
| sftv2_T1.2 | 3% | 11% | 14% | 27% | 39% |
| sftv2_T1.5 | 0% | 8% | 17% | 38% | 50% |

## B. Exploration/exploitation: SFT base vs sparse vs diversity-preserving GRPO (seeded)

| model | in-dist | bv@4 | bv@8 | bv@16 |
|---|---|---|---|---|
| sftv2base | 27% | 7% | 13% | 21% |
| control40 | 27% | 4% | 8% | 12% |
| control80 | 42% | 2% | 2% | 5% |
| divA20 | 31% | 7% | 14% | 18% |
| divB20 | 32% | 6% | 13% | 16% |
| divC20 | 31% | 11% | 12% | 18% |
| divD20 | 26% | 3% | 7% | 13% |
