"""
Preprocess dataset for 24-point game task.

Both nlile/24-game and test-time-compute/game-of-24 contain the exact same
1362 solvable 4-number puzzles. We split them by difficulty (Rank):
  Train      : Rank 1  – 1262 (easiest, high solved-rate)
  Test (OOD) : Rank 1263 – 1362 (hardest 100, lowest solved-rate)
  Hallucination: synthetically generated unsolvable 4-tuples from {1..13}
                 (nlile/24-game no longer contains solvable=False rows)

Usage:
  python examples/data_preprocess/game24.py --local_dir ~/data/game24
"""

import os
import re
import random
import argparse
from itertools import combinations_with_replacement
from datasets import load_dataset, Dataset


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #

def make_prefix(numbers: list, template_type: str = "qwen-instruct") -> str:
    nums_str = str(numbers)
    if template_type == "base":
        return (
            f"A conversation between User and Assistant. The user asks a question, "
            f"and the Assistant solves it. The assistant first thinks about the reasoning "
            f"process in the mind and then provides the user with the answer.\n"
            f"User: Using the numbers {nums_str}, create an equation that equals 24. "
            f"You can use basic arithmetic operations (+, -, *, /) and each number must "
            f"be used exactly once. Show your work in <think> </think> tags. "
            f"Return the final answer in <answer> </answer> tags, "
            f"for example <answer> (1 + 2) * 8 </answer>.\n"
            f"Assistant: Let me solve this step by step.\n<think>"
        )
    else:  # qwen-instruct
        # Use Qwen2.5's DEFAULT system prompt so the GRPO/eval prompt rendering matches the
        # SFT data exactly. SFTDataset applies the chat template to a bare user turn, which
        # injects this default system message; aligning here lets the SFT warm-start transfer
        # cleanly into GRPO (and keeps both comparison arms on an identical prompt).
        return (
            f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
            f"assistant.<|im_end|>\n<|im_start|>user\nUsing the numbers {nums_str}, create an "
            f"equation that equals 24. You can use basic arithmetic operations "
            f"(+, -, *, /) and each number must be used exactly once. "
            f"Show your work in <think> </think> tags. "
            f"Return the final answer in <answer> </answer> tags, "
            f"for example <answer> (1 + 2) * 8 </answer>.<|im_end|>\n"
            f"<|im_start|>assistant\nLet me solve this step by step.\n<think>"
        )


def build_sample(numbers: list, split: str, idx: int,
                 solvable: bool, template_type: str, solved_rate: float = None):
    prompt = make_prefix(numbers, template_type)
    sample = {
        "data_source": "game24",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            # solved_rate is copied into ground_truth so the reward function can
            # difficulty-weight (reward *= 1 + (1 - solved_rate)) without extra plumbing.
            "ground_truth": {"numbers": numbers, "target": 24,
                             "solved_rate": float(solved_rate) if solved_rate is not None else 1.0},
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "solvable": solvable,
        },
    }
    if solved_rate is not None:
        sample["extra_info"]["solved_rate"] = solved_rate
    return sample


# --------------------------------------------------------------------------- #
# Generate synthetic unsolvable 4-tuples from {1..13}
# --------------------------------------------------------------------------- #

def generate_unsolvable(solvable_set: set, n: int = 100, seed: int = 42) -> list:
    """Return n sorted tuples from {1..13}^4 that are NOT in solvable_set."""
    all_combos = list(combinations_with_replacement(range(1, 14), 4))
    unsolvable = [c for c in all_combos if c not in solvable_set]
    print(f"  Total 4-tuples from {{1..13}}: {len(all_combos)}")
    print(f"  Solvable: {len(solvable_set)}, Unsolvable: {len(unsolvable)}")
    random.seed(seed)
    return random.sample(unsolvable, min(n, len(unsolvable)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/game24")
    parser.add_argument("--template_type", default="qwen-instruct",
                        choices=["base", "qwen-instruct"])
    parser.add_argument("--train_size", type=int, default=1262,
                        help="Number of easiest puzzles for training (Rank 1..N)")
    parser.add_argument("--test_size", type=int, default=100,
                        help="Number of hardest puzzles for OOD testing")
    parser.add_argument("--halluc_size", type=int, default=100,
                        help="Number of synthetic unsolvable puzzles for hallucination test")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load test-time-compute/game-of-24 (has Rank / difficulty)
    # ------------------------------------------------------------------ #
    print("Loading test-time-compute/game-of-24 ...")
    try:
        ds = load_dataset("test-time-compute/game-of-24", split="train")
    except Exception:
        ds = load_dataset("test-time-compute/game-of-24", split="test")
    print(f"  columns: {ds.column_names}")
    print(f"  total rows: {len(ds)}")

    def parse_nums(row):
        return [int(x) for x in re.findall(r"\d+", row["Puzzles"])]

    def parse_solved_rate(row):
        sr = row.get("Solved rate", "0%")
        return float(str(sr).replace("%", "").strip()) / 100.0

    # Sort by Rank ascending: Rank 1 = easiest, Rank 1362 = hardest
    rows = sorted(ds, key=lambda x: x["Rank"])

    total = len(rows)
    assert args.train_size + args.test_size <= total, (
        f"train_size({args.train_size}) + test_size({args.test_size}) > total({total})"
    )

    train_rows = rows[:args.train_size]
    test_rows  = rows[total - args.test_size:]

    print(f"\n  Train : Rank {train_rows[0]['Rank']}–{train_rows[-1]['Rank']}, "
          f"solved_rate {parse_solved_rate(train_rows[0]):.1%}–{parse_solved_rate(train_rows[-1]):.1%}")
    print(f"  Test  : Rank {test_rows[0]['Rank']}–{test_rows[-1]['Rank']}, "
          f"solved_rate {parse_solved_rate(test_rows[0]):.1%}–{parse_solved_rate(test_rows[-1]):.1%}")

    train_samples = [
        build_sample(parse_nums(r), "train", i, True,
                     args.template_type, parse_solved_rate(r))
        for i, r in enumerate(train_rows)
    ]
    test_samples = [
        build_sample(parse_nums(r), "test", i, True,
                     args.template_type, parse_solved_rate(r))
        for i, r in enumerate(test_rows)
    ]

    # Leakage guard: train and test puzzles (as sorted number-multisets) must be disjoint.
    train_keys = set(tuple(sorted(parse_nums(r))) for r in train_rows)
    test_keys = set(tuple(sorted(parse_nums(r))) for r in test_rows)
    overlap = train_keys & test_keys
    assert not overlap, f"TRAIN/TEST LEAKAGE: {len(overlap)} shared number-multisets, e.g. {list(overlap)[:5]}"
    print(f"  Leakage check OK: train({len(train_keys)}) ∩ test({len(test_keys)}) = 0")

    # ------------------------------------------------------------------ #
    # Hallucination test: synthetic unsolvable 4-tuples
    # ------------------------------------------------------------------ #
    print("\nGenerating hallucination test set (unsolvable puzzles) ...")
    solvable_set = set(tuple(sorted(parse_nums(r))) for r in rows)
    unsolvable_tuples = generate_unsolvable(solvable_set, args.halluc_size)
    halluc_samples = [
        build_sample(list(t), "hallucination", i, False, args.template_type)
        for i, t in enumerate(unsolvable_tuples)
    ]

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    def to_parquet(samples, name):
        if not samples:
            print(f"  Skipping {name}.parquet (0 samples)")
            return
        path = os.path.join(local_dir, f"{name}.parquet")
        Dataset.from_list(samples).to_parquet(path)
        print(f"  Saved {len(samples)} samples → {path}")

    print()
    to_parquet(train_samples,  "train")
    to_parquet(test_samples,   "test")
    to_parquet(halluc_samples, "hallucination")

    print("\nDone. Dataset summary:")
    print(f"  train         : {len(train_samples)} (Rank 1–{args.train_size}, easy)")
    print(f"  test (OOD)    : {len(test_samples)}  (Rank {total-args.test_size+1}–{total}, hard)")
    print(f"  hallucination : {len(halluc_samples)} (synthetic unsolvable)")


if __name__ == "__main__":
    main()
