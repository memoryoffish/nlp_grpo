"""
Assignment-compliant Game-of-24 data prep.

- TRAIN : nlile/24-game (solvable=True), EXCLUDING the test puzzles  -> 1262 puzzles.
- TEST  : test-time-compute/game-of-24 indices 900-999 (Rank 901-1000), the Tree-of-Thoughts
          paper's standard 100 hard puzzles.
- HALLUC: synthetic unsolvable 4-tuples from {1..13} (nlile no longer exposes solvable=False).

This replaces the earlier (non-compliant) split that trained on Rank 1-1262 and tested on the
absolute-hardest Rank 1263-1362 (which leaked: the ToT test puzzles were inside that train set).

Usage:
  python examples/data_preprocess/game24_official.py --local_dir data/game24_official
"""
import argparse
import os
import re
import sys

from datasets import load_dataset, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from game24 import make_prefix, build_sample, generate_unsolvable  # reuse


def ms(nums):
    return tuple(sorted(int(x) for x in nums))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="data/game24_official")
    ap.add_argument("--template_type", default="qwen-instruct")
    ap.add_argument("--test_lo", type=int, default=900)   # 0-indexed inclusive
    ap.add_argument("--test_hi", type=int, default=1000)  # exclusive -> 900..999 = 100 puzzles
    ap.add_argument("--halluc_size", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)

    # ---- TEST: ttc indices 900-999 (ToT hard set) ----
    print("Loading test-time-compute/game-of-24 ...")
    ttc = load_dataset("test-time-compute/game-of-24", split="train")
    test_rows = []
    for i in range(args.test_lo, args.test_hi):
        nums = [int(x) for x in re.findall(r"\d+", ttc[i]["Puzzles"])][:4]
        sr = float(str(ttc[i]["Solved rate"]).replace("%", "").strip()) / 100.0
        test_rows.append((nums, sr))
    test_set = set(ms(n) for n, _ in test_rows)
    print(f"  test = ttc idx {args.test_lo}-{args.test_hi-1}: {len(test_rows)} puzzles, unique {len(test_set)}")

    # ---- TRAIN: nlile solvable minus test ----
    print("Loading nlile/24-game ...")
    nlile = load_dataset("nlile/24-game", split="train")
    train_rows, all_solvable = [], set()
    for r in nlile:
        nums = list(r["numbers"])
        if not bool(r.get("solvable", True)):
            continue
        all_solvable.add(ms(nums))
        if ms(nums) in test_set:
            continue
        train_rows.append((nums, float(r.get("solved_rate", 0.0) or 0.0)))
    print(f"  train = nlile solvable minus test: {len(train_rows)} puzzles")

    # leakage assertion
    train_set = set(ms(n) for n, _ in train_rows)
    assert train_set.isdisjoint(test_set), "LEAKAGE: train overlaps test!"
    print(f"  leakage check: train ∩ test = {len(train_set & test_set)} (must be 0)")

    # ---- HALLUC: synthetic unsolvable (exclude all solvable + test) ----
    halluc = generate_unsolvable(all_solvable | test_set, n=args.halluc_size)
    print(f"  hallucination = {len(halluc)} synthetic unsolvable tuples")

    tt = args.template_type
    train_samples = [build_sample(n, "train", i, True, tt, sr) for i, (n, sr) in enumerate(train_rows)]
    test_samples = [build_sample(n, "test", i, True, tt, sr) for i, (n, sr) in enumerate(test_rows)]
    halluc_samples = [build_sample(list(t), "hallucination", i, False, tt) for i, t in enumerate(halluc)]

    def save(samples, name):
        path = os.path.join(args.local_dir, f"{name}.parquet")
        Dataset.from_list(samples).to_parquet(path)
        print(f"  saved {len(samples):4d} -> {path}")

    save(train_samples, "train")
    save(test_samples, "test")
    save(halluc_samples, "hallucination")
    print("done.")


if __name__ == "__main__":
    main()
