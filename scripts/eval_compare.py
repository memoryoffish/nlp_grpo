"""
Comparison evaluation for the SFT->GRPO vs pure-GRPO study.

Loads each model ONCE and evaluates it on several parquet test sets:
  - game24 hard test  (in-distribution-ish, hardest 100 puzzles)  -> greedy acc + pass@k
  - hallucination     (unsolvable puzzles) -> "solved" rate (LOWER is better; fabrication)
  - countdown OOD     (3-4 numbers, any target) -> greedy acc (generalization, bonus)

Emits a markdown table + a JSON dump.

Usage:
  python scripts/eval_compare.py \
      --model base:/mnt/.../Qwen2.5-1.5B-Instruct \
      --model SFT:checkpoints/TinyZero/game24-sft/global_step_111 \
      --model pure-GRPO:checkpoints/TinyZero/game24-grpo-base/actor/global_step_160 \
      --model SFT-GRPO:checkpoints/TinyZero/game24-grpo-sftinit/actor/global_step_160 \
      --out_md results/comparison.md --out_json results/comparison.json
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verl.utils.reward_score import game24, countdown  # noqa: E402


extract_answer = game24.extract_answer  # exact-verifier answer extractor


def _score_fn(data_source):
    if "countdown" in data_source:
        return countdown.compute_score
    return game24.compute_score


def build_prompt(row):
    pf = row["prompt"]
    if hasattr(pf, "tolist"):
        pf = pf.tolist()
    if isinstance(pf, list):
        for msg in reversed(pf):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg["content"]
        return pf[-1]["content"]
    return str(pf)


def gt_of(row):
    rm = row["reward_model"]
    return rm["ground_truth"] if isinstance(rm, dict) else rm.get("ground_truth", {})


@torch.inference_mode()
def eval_dataset(model, tok, df, data_source, batch_size, max_new_tokens, kmax=1,
                 temperature=1.0, frontier=(1, 2, 4, 8, 16, 32, 64)):
    """Greedy accuracy + the verifier best-of-N frontier.

    best_verified@k = fraction of puzzles where AT LEAST ONE of the first k sampled
    candidates is verified correct (==24 with the right numbers). Because we have an
    EXACT verifier, this is a legitimate deployable solve rate (sample k, output the
    verified one). maj@kmax = majority-vote expression is correct.
    """
    from collections import Counter
    score_fn = _score_fn(data_source)
    greedy_scores = []
    sample_correct = []   # per puzzle: list[bool] length kmax
    sample_exprs = []     # per puzzle: list[str|None] length kmax
    for i in tqdm(range(0, len(df), batch_size), desc=f"{data_source}", leave=False):
        batch = df.iloc[i:i + batch_size]
        prompts = [build_prompt(r) for _, r in batch.iterrows()]
        gts = [gt_of(r) for _, r in batch.iterrows()]
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        ilen = inputs["input_ids"].shape[1]
        # greedy 1-shot
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            gen = tok.decode(out[j][ilen:], skip_special_tokens=False)
            greedy_scores.append(score_fn(solution_str=prompts[j] + gen, ground_truth=gts[j]))
        # sample kmax candidates in one call (ordered per-prompt consecutive)
        if kmax > 1:
            so = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                temperature=temperature, top_p=0.95,
                                num_return_sequences=kmax, pad_token_id=tok.eos_token_id)
            for j in range(len(batch)):
                flags, exprs = [], []
                for r in range(kmax):
                    g = tok.decode(so[j * kmax + r][ilen:], skip_special_tokens=False)
                    full = prompts[j] + g
                    flags.append(score_fn(solution_str=full, ground_truth=gts[j]) >= 1.0)
                    exprs.append(extract_answer(full))
                sample_correct.append(flags)
                sample_exprs.append(exprs)
    n = len(greedy_scores)
    res = {
        "n": n,
        "mean_score": round(sum(greedy_scores) / n, 4),
        "greedy_acc": round(sum(1 for s in greedy_scores if s >= 1.0) / n, 4),
        "n_correct": sum(1 for s in greedy_scores if s >= 1.0),
    }
    if kmax > 1:
        for k in frontier:
            if k <= kmax:
                res[f"best_verified@{k}"] = round(sum(1 for fl in sample_correct if any(fl[:k])) / n, 4)
        # majority-vote: most common answer expression; correct iff that expression verifies
        maj = 0
        for fl, ex in zip(sample_correct, sample_exprs):
            cnt = Counter(e for e in ex if e)
            if not cnt:
                continue
            top = cnt.most_common(1)[0][0]
            for e, f in zip(ex, fl):
                if e == top:
                    maj += 1 if f else 0
                    break
        res[f"maj@{kmax}"] = round(maj / n, 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="label:path")
    ap.add_argument("--test", default="data/game24/test.parquet")
    ap.add_argument("--train", default="data/game24/train.parquet",
                    help="in-distribution (training-distribution) eval; sampled")
    ap.add_argument("--train_n", type=int, default=100, help="subsample of train for in-dist eval")
    ap.add_argument("--halluc", default="data/game24/hallucination.parquet")
    ap.add_argument("--countdown", default="data/countdown/test.parquet")
    ap.add_argument("--countdown_n", type=int, default=256, help="subsample countdown for speed")
    ap.add_argument("--bon", "--passk", dest="bon", type=int, default=16,
                    help="max samples for the verifier best-of-N frontier on the hard test")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no_halluc", action="store_true", help="skip hallucination eval (fast screening)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--out_md", default="results/comparison.md")
    ap.add_argument("--out_json", default="results/comparison.json")
    args = ap.parse_args()

    # Eval always uses the SPARSE reward so accuracy/mean are clean (==24 only); shaping
    # never affects correctness (correct==1.0 in both modes) but would change mean_score.
    os.environ["GAME24_REWARD"] = "sparse"

    # Seed for reproducibility. best-of-N is a single stochastic draw at n=100 and is
    # upward-biased / high-variance (~+/-10pp at bv@64); seeding makes runs comparable.
    import random as _random
    torch.manual_seed(args.seed); _random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    test_df = pd.read_parquet(args.test)
    halluc_df = pd.read_parquet(args.halluc)
    cd_df = pd.read_parquet(args.countdown)
    if args.countdown_n and len(cd_df) > args.countdown_n:
        cd_df = cd_df.iloc[:args.countdown_n]
    train_df = pd.read_parquet(args.train)
    if args.train_n and len(train_df) > args.train_n:
        train_df = train_df.iloc[:args.train_n]  # deterministic head sample (in-distribution)

    # Leakage guard at eval time too: train vs test number-multisets disjoint.
    def _keys(d): return set(tuple(sorted(int(x) for x in gt_of(r)["numbers"])) for _, r in d.iterrows())
    leak = _keys(pd.read_parquet(args.test)) & _keys(pd.read_parquet(args.train))
    assert not leak, f"TRAIN/TEST LEAKAGE at eval: {len(leak)} shared, e.g. {list(leak)[:5]}"
    print(f"[leakage check] train vs test disjoint OK")

    table = {}
    for spec in args.model:
        label, _, path = spec.partition(":")
        print(f"\n===== {label}  ({path}) =====")
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).cuda().eval()
        row = {}
        # always: hard-test (greedy + verifier best-of-N) — the screening metric
        row["test"] = eval_dataset(model, tok, test_df, "game24", args.batch_size,
                                   args.max_new_tokens, kmax=args.bon, temperature=args.temperature)
        # optional (skip with *_n=0 for fast screening): in-dist, hallucination, countdown OOD
        if args.train_n != 0:
            row["train_indist"] = eval_dataset(model, tok, train_df, "game24", args.batch_size,
                                                args.max_new_tokens, kmax=1)
        if not args.no_halluc:
            row["hallucination"] = eval_dataset(model, tok, halluc_df, "game24", args.batch_size,
                                                args.max_new_tokens, kmax=1)
        if args.countdown_n != 0:
            row["countdown_ood"] = eval_dataset(model, tok, cd_df, "countdown", args.batch_size,
                                                args.max_new_tokens, kmax=1)
        table[label] = row
        print(json.dumps(row, indent=2))
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(table, f, indent=2)

    # markdown table: greedy + verifier best-of-N frontier on the hard test.
    # Frontier columns are DYNAMIC: only the k actually computed (k <= bon), deduped —
    # avoids duplicate/zero columns when bon is small.
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.bon]
    bv = lambda r, k: r['test'].get(f'best_verified@{k}', 0)
    pct = lambda x: "-" if (x != x) else f"{x:.0%}"   # "-" for NaN (skipped dataset)
    g = lambda r, k: r.get(k, {}).get('greedy_acc', float('nan'))
    bv_hdr = " | ".join(f"bv@{k}" for k in ks)
    bv_sep = "|".join(["---"] * len(ks))
    lines = [
        "Metric defs: greedy = 1-shot deterministic; best_verified@k = sample k, output the "
        "verifier-confirmed-correct one (deployable, exact verifier); maj@k = majority-vote answer.",
        "",
        f"| Model | train(in-dist) greedy | hard-test greedy | {bv_hdr} | maj@{args.bon} | halluc↓ | Countdown OOD |",
        f"|---|---|---|{bv_sep}|---|---|---|",
    ]
    for label, r in table.items():
        bv_cells = " | ".join(pct(bv(r, k)) for k in ks)
        lines.append(
            f"| {label} | {pct(g(r,'train_indist'))} | {pct(r['test']['greedy_acc'])} | "
            f"{bv_cells} | {pct(r['test'].get(f'maj@{args.bon}',0))} | "
            f"{pct(g(r,'hallucination'))} | {pct(g(r,'countdown_ood'))} |"
        )
    md = "\n".join(lines)
    with open(args.out_md, "w") as f:
        f.write(md + "\n")
    print("\n" + md)
    print(f"\nwrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
