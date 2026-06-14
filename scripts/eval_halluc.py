"""
Proper hallucination evaluation on UNSOLVABLE puzzles.

The assignment asks: does the model fabricate an expression for puzzles that have NO solution?
Because the R1 output format forces an <answer>, every model emits a (necessarily wrong)
expression -> the raw "fabrication rate" is ~100% and uninformative on its own. The interesting
question is whether the verify-by-compute SELF-CRITIC can REJECT these fabrications, i.e. abstain.

Per model, on the synthetic unsolvable set:
  fab_greedy        : fraction where greedy emits a 4-number expression claiming a solution
  critic_accept_rate: mean fraction of k sampled candidates the self-critic (wrongly) marks CORRECT
                      (false-accept rate; puzzle is unsolvable so any accept is a hallucination)
  abstain_rate      : fraction of puzzles where the self-critic marks ALL k candidates WRONG
                      -> the model correctly refuses to claim a solution (hallucination DEFENDED)

Usage: python scripts/eval_halluc.py --model L:PATH [...] --halluc data/game24_official/hallucination.parquet --k 8
"""
import argparse, json, os, sys, importlib.util
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verl.utils.reward_score.game24 import extract_answer  # noqa: E402
spec = importlib.util.spec_from_file_location("cc", os.path.join(os.path.dirname(__file__), "eval_selfcritic_compute.py"))
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)  # reuse critic_judge / build_prompt / gt_of / sample_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--halluc", default="data/game24_official/hallucination.parquet")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--out_json", default="results/halluc_selfcritic.json")
    ap.add_argument("--out_md", default="results/halluc_selfcritic.md")
    args = ap.parse_args()
    df = pd.read_parquet(args.halluc)
    table = {}
    for spec_m in args.model:
        label, _, path = spec_m.partition(":")
        print(f"\n===== {label} ({path}) =====", flush=True)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).cuda().eval()
        n = len(df); fab_greedy = 0; accept_sum = 0.0; accept_den = 0; abstain = 0
        for _, row in tqdm(df.iterrows(), total=n, desc=label):
            prompt = cc.build_prompt(row); gt = cc.gt_of(row)
            penc = tok(prompt, return_tensors="pt").to(model.device)
            gout = model.generate(**penc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                  pad_token_id=tok.eos_token_id)
            gtxt = tok.decode(gout[0][penc["input_ids"].shape[1]:], skip_special_tokens=False)
            fab_greedy += 1 if (extract_answer(prompt + gtxt) or "") else 0
            samples = cc.sample_candidates(model, tok, prompt, args.k, args.temperature, args.max_new_tokens)
            exprs = [extract_answer(prompt + s) or "" for s in samples]
            cand = [e for e in exprs if e]
            if not cand:
                abstain += 1  # emitted no parseable expression at all -> abstains
                continue
            verdicts = cc.critic_judge(model, tok, gt["numbers"], cand, max_new_tokens=100)
            acc = sum(1 for v in verdicts if v)  # critic-accepted (false accepts)
            accept_sum += acc; accept_den += len(verdicts)
            if acc == 0:
                abstain += 1  # critic rejects every candidate -> model abstains (defended)
        res = {"n": n,
               "fab_greedy_rate": round(fab_greedy / n, 4),
               "critic_accept_rate": round(accept_sum / accept_den, 4) if accept_den else 0.0,
               "abstain_rate": round(abstain / n, 4)}
        table[label] = res
        print(json.dumps(res, indent=2), flush=True)
        del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(table, open(args.out_json, "w"), indent=2)
    lines = ["Hallucination on UNSOLVABLE puzzles: raw fabrication vs self-critic abstention", "",
             "| Model | greedy 瞎编率 | 自评误受率(越低越好) | 自评弃答率(越高越好) |", "|---|---|---|---|"]
    for lb, r in table.items():
        lines.append(f"| {lb} | {r['fab_greedy_rate']:.0%} | {r['critic_accept_rate']:.0%} | {r['abstain_rate']:.0%} |")
    open(args.out_md, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
