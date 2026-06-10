# Qualitative rollout samples during GRPO training (reward-fn debug prints)

## Arm A — pure GRPO (base init): parrots prompt example / wrong numbers
[0m Numbers: [ 4  5  6 11] | Target: 24
[0m Extracted expr: None
[0m Numbers: [ 1  4  9 13] | Target: 24
[0m Extracted expr: (1 + 4) * 9 - 13
[0m Wrong result: (1 + 4) * 9 - 13 = 32.0 (expected 24) → 0.1
[0m Numbers: [ 3  3  5 10] | Target: 24
[0m Extracted expr: None
[0m Numbers: [ 5  6  9 13] | Target: 24
[0m Extracted expr: None
[0m Numbers: [ 5  7  7 10] | Target: 24
[0m Extracted expr: (7-10)/7 * (5+7)
[0m Wrong numbers in expr → 0.1
[0m Numbers: [ 5  6 10 10] | Target: 24
[0m Extracted expr: None
[0m Numbers: [ 6  6  9 11] | Target: 24
[0m Extracted expr: None
[0m Numbers: [3 3 4 4] | Target: 24
[0m Extracted expr: (4 * 4 - 3) * 3
[0m Wrong result: (4 * 4 - 3) * 3 = 39.0 (expected 24) → 0.1
[0m Numbers: [2 4 4 9] | Target: 24
[0m Extracted expr: (1 + 2) * 8 = 24
[0m Wrong numbers in expr → 0.1
[0m Numbers: [ 2  8 12 12] | Target: 24
[0m Extracted expr: 8 * 12 - 2

## Arm B — SFT->GRPO: uses correct numbers, frequently correct
[0m Numbers: [ 2  3 13 13] | Target: 24
[0m Extracted expr: 2 * 13 - 3 + 13
[0m Wrong result: 2 * 13 - 3 + 13 = 36.0 (expected 24) → 0.1
[0m Numbers: [ 2  7  7 10] | Target: 24
[0m Extracted expr: (7 - 2) * 7 - 10
[0m Wrong result: (7 - 2) * 7 - 10 = 25.0 (expected 24) → 0.1
[0m Numbers: [3 3 5 7] | Target: 24
[0m Extracted expr: (3 + 5) * 3 + 7
[0m Wrong result: (3 + 5) * 3 + 7 = 31.0 (expected 24) → 0.1
[0m Numbers: [ 1  5  9 13] | Target: 24
[0m Extracted expr: (9 - 1 * 5) * 13
[0m Wrong result: (9 - 1 * 5) * 13 = 52.0 (expected 24) → 0.1
[0m Numbers: [ 2  3  3 11] | Target: 24
[0m Extracted expr: 3 * 11 - 2 - 3
[0m Wrong result: 3 * 11 - 2 - 3 = 28.0 (expected 24) → 0.1
[0m Numbers: [ 1  5  5 11] | Target: 24
[0m Extracted expr: 1 + 5 + 5 + 11
[0m Wrong result: 1 + 5 + 5 + 11 = 22.0 (expected 24) → 0.1
[0m Numbers: [ 1  1 11 13] | Target: 24
[0m Extracted expr: (11 + 13) * (1 - 1)
[0m Wrong result: (11 + 13) * (1 - 1) = 0.0 (expected 24) → 0.1
[0m Numbers: [1 1 5 8] | Target: 24
[0m Extracted expr: 1 * 1 + 5 + 8
[0m Wrong result: 1 * 1 + 5 + 8 = 14.0 (expected 24) → 0.1
