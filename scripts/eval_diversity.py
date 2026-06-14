"""
Diagnose WHY GRPO raises greedy but lowers best-of-N / self-critic: sampling-diversity collapse.

For each model, sample k candidates per test puzzle (T=1.0) and measure:
  uniq        : mean # of DISTINCT non-empty expressions among the k samples (diversity)
  uniq_correct: mean # of DISTINCT *correct* expressions (what best-of-N / self-critic feed on)
  collapse    : fraction of puzzles with <=2 distinct expressions (heavy mode collapse)
  greedy_acc / bv@k for reference.
Also dumps, for a few puzzles, the full candidate set (expr + correctness) as concrete examples.

Usage: python scripts/eval_diversity.py --model L:PATH [...] --test data/game24_official/test.parquet --k 16 --examples 3
"""
import argparse, json, os, sys
from collections import Counter
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verl.utils.reward_score.game24 import compute_score, extract_answer  # noqa: E402


def build_prompt(row):
    pf = row["prompt"]
    if hasattr(pf, "tolist"):
        pf = pf.tolist()
    if isinstance(pf, list):
        for m in reversed(pf):
            if isinstance(m, dict) and m.get("role") == "user":
                return m["content"]
    return str(pf)


def gt_of(row):
    rm = row["reward_model"]
    return rm["ground_truth"] if isinstance(rm, dict) else rm.get("ground_truth", {})


@torch.inference_mode()
def sample_candidates(model, tok, prompt, k, temperature, max_new_tokens):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                         num_return_sequences=k, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.eos_token_id)
    ilen = enc["input_ids"].shape[1]
    return [tok.decode(out[j][ilen:], skip_special_tokens=False) for j in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--test", default="data/game24_official/test.parquet")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out_json", default="results/diversity.json")
    args = ap.parse_args()
    df = pd.read_parquet(args.test)
    table = {}
    for spec in args.model:
        label, _, path = spec.partition(":")
        print(f"\n===== {label} ({path}) =====", flush=True)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).cuda().eval()
        n = len(df); uniq = uniq_c = collapse = bv = 0; examples = []
        for pi, (_, row) in enumerate(tqdm(df.iterrows(), total=n, desc=label)):
            prompt = build_prompt(row); gt = gt_of(row)
            s = sample_candidates(model, tok, prompt, args.k, args.temperature, args.max_new_tokens)
            exprs = [extract_answer(prompt + x) or "" for x in s]
            corr = [compute_score(prompt + x, gt) >= 1.0 for x in s]
            ne = [e for e in exprs if e]
            U = set(ne); Uc = set(e for e, c in zip(exprs, corr) if e and c)
            uniq += len(U); uniq_c += len(Uc); bv += 1 if any(corr) else 0
            collapse += 1 if len(U) <= 2 else 0
            if pi < args.examples:
                cnt = Counter(ne)
                examples.append({"numbers": [int(x) for x in gt["numbers"]],
                                 "distinct": [{"expr": e, "n": cnt[e], "correct": e in Uc}
                                              for e in sorted(cnt, key=lambda z: -cnt[z])]})
        res = {"n": n, "k": args.k,
               "uniq_mean": round(uniq / n, 2),
               "uniq_correct_mean": round(uniq_c / n, 2),
               "collapse_frac(<=2 uniq)": round(collapse / n, 3),
               "bv@%d" % args.k: round(bv / n, 3),
               "examples": examples}
        table[label] = res
        print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2), flush=True)
        del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(table, open(args.out_json, "w"), indent=2, ensure_ascii=False, default=int)
    print("\nsaved", args.out_json)


if __name__ == "__main__":
    main()
