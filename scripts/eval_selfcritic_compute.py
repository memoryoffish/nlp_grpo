"""
Strongest self-critic: few-shot + FORCED step-by-step computation (verify-by-compute).

Diagnosis from the audit: the 1.5B model, when judging, does NOT compute — it rubber-stamps
any structurally-plausible expression as 'sure' (it called (4-5+10)*6=54 "sure"). The two
earlier ToT-faithful / CoT critics failed for this reason.

This critic combines BOTH missing ingredients:
  (1) few-shot examples that include WRONG answers labelled WRONG (anti yes-bias), AND
  (2) each example FIRST computes the numeric value step by step, THEN judges — so the final
      decision reduces to the trivial check "is the computed value exactly 24?".
This is the faithful "model must compute the answer" critic the task calls for.

Reports greedy / selfcritic_cmp@k / best_verified@k / maj@k + verifier precision/recall.

Usage:
  python scripts/eval_selfcritic_compute.py --model L:PATH [...] --test data/game24_official/test.parquet --k 16
"""
import argparse, json, os, sys, re
from collections import Counter
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verl.utils.reward_score.game24 import compute_score, extract_answer  # noqa: E402

# few-shot: compute the value FIRST, then judge; includes correct + wrong worked examples.
# NOTE: fed as a RAW completion prompt (NOT chat template). The SFT/GRPO models are locked into
# their R1 "solve" format and degenerate into "Let me solve this step by step." loops when given a
# chat-template critic prompt; raw few-shot completion makes them continue the judging pattern.
CRITIC_HEAD = ("Check each 24-game expression. First compute its value step by step, then judge: "
               "CORRECT only if it uses each input exactly once AND the value is exactly 24, else WRONG.\n\n")
FEWSHOT = """Input: 4 4 6 8
Expression: (4 + 8) * (6 - 4)
Compute: 4 + 8 = 12; 6 - 4 = 2; 12 * 2 = 24. Value = 24. Numbers used: 4,8,6,4 = the four inputs, each once.
Judge: CORRECT

Input: 4 5 6 10
Expression: 4 * 10 - 5 - 6
Compute: 4 * 10 = 40; 40 - 5 = 35; 35 - 6 = 29. Value = 29.
Judge: WRONG

Input: 1 2 4 7
Expression: (1 + 2 + 7) * 4
Compute: 1 + 2 + 7 = 10; 10 * 4 = 40. Value = 40.
Judge: WRONG

Input: 2 5 8 11
Expression: 2 * (5 + 8) - 11
Compute: 5 + 8 = 13; 2 * 13 = 26; 26 - 11 = 15. Value = 15.
Judge: WRONG

Input: 2 9 10 12
Expression: 2 * 12 * (10 - 9)
Compute: 10 - 9 = 1; 2 * 12 = 24; 24 * 1 = 24. Value = 24. Numbers used: 2,12,10,9 = the four inputs, each once.
Judge: CORRECT
"""

_VERDICT_RE = re.compile(r"Judge:\s*\**\s*(CORRECT|WRONG)", re.I)
_FALLBACK_RE = re.compile(r"\b(CORRECT|WRONG)\b", re.I)


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


def critic_prompt(tok, numbers, expr):
    in_str = " ".join(str(int(x)) for x in numbers)
    # RAW completion (no chat template): few-shot pattern the model continues.
    return CRITIC_HEAD + FEWSHOT + f"\nInput: {in_str}\nExpression: {expr if expr else '(none)'}\nCompute:"


def parse_verdict(text):
    # take the FIRST verdict: the model often fabricates extra Input/Expression/Judge blocks after.
    m = _VERDICT_RE.findall(text) or _FALLBACK_RE.findall(text)
    return (m[0].upper() == "CORRECT") if m else None


@torch.inference_mode()
def critic_judge(model, tok, numbers, exprs, max_new_tokens=96, batch_size=16):
    verdicts = []
    for i in range(0, len(exprs), batch_size):
        chunk = exprs[i:i + batch_size]
        prompts = [critic_prompt(tok, numbers, e) for e in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id)
        ilen = enc["input_ids"].shape[1]
        for r in range(len(chunk)):
            verdicts.append(parse_verdict(tok.decode(out[r][ilen:], skip_special_tokens=True)))
    return verdicts


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
    ap.add_argument("--critic_max_new_tokens", type=int, default=96)
    ap.add_argument("--out_md", default="results/selfcritic_compute.md")
    ap.add_argument("--out_json", default="results/selfcritic_compute.json")
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
        n = len(df); greedy_c = sc_c = bv_c = maj_c = 0; TP = FP = FN = TN = 0
        for _, row in tqdm(df.iterrows(), total=n, desc=label):
            prompt = build_prompt(row); gt = gt_of(row)
            penc = tok(prompt, return_tensors="pt").to(model.device)
            gout = model.generate(**penc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                  pad_token_id=tok.eos_token_id)
            gtxt = tok.decode(gout[0][penc["input_ids"].shape[1]:], skip_special_tokens=False)
            greedy_c += compute_score(prompt + gtxt, gt) >= 1.0

            samples = sample_candidates(model, tok, prompt, args.k, args.temperature, args.max_new_tokens)
            exprs = [extract_answer(prompt + s) or "" for s in samples]
            corr = [compute_score(prompt + s, gt) >= 1.0 for s in samples]
            bv_c += any(corr)
            valid = [e for e in exprs if e]
            if valid:
                top = Counter(valid).most_common(1)[0][0]
                maj_c += compute_score(prompt + f"<answer>{top}</answer>", gt) >= 1.0
            cand_idx = [j for j, e in enumerate(exprs) if e]
            if cand_idx:
                verdicts = critic_judge(model, tok, gt["numbers"], [exprs[j] for j in cand_idx],
                                        max_new_tokens=args.critic_max_new_tokens)
                for jj, v in zip(cand_idx, verdicts):
                    if v is None:
                        continue
                    a = corr[jj]
                    if v and a: TP += 1
                    elif v and not a: FP += 1
                    elif (not v) and a: FN += 1
                    else: TN += 1
                pick = next((jj for jj, v in zip(cand_idx, verdicts) if v), cand_idx[0])
                sc_c += corr[pick]
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        vacc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0.0
        res = {"n": n, "greedy": round(greedy_c / n, 4),
               f"selfcritic_cmp@{args.k}": round(sc_c / n, 4),
               f"best_verified@{args.k}": round(bv_c / n, 4),
               f"maj@{args.k}": round(maj_c / n, 4),
               "verifier_precision": round(prec, 4), "verifier_recall": round(rec, 4),
               "verifier_accuracy": round(vacc, 4),
               "confusion": {"TP": TP, "FP": FP, "FN": FN, "TN": TN}}
        table[label] = res
        print(json.dumps(res, indent=2), flush=True)
        del model; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(table, open(args.out_json, "w"), indent=2)
    k = args.k
    lines = [f"Verify-by-compute self-critic (few-shot + forced computation, k={k}); test={args.test}", "",
             f"| Model | greedy | selfcritic_cmp@{k} | best_verified@{k} | maj@{k} | verif_prec | verif_rec | verif_acc |",
             "|---|---|---|---|---|---|---|---|"]
    for lb, r in table.items():
        lines.append(f"| {lb} | {r['greedy']:.2%} | {r[f'selfcritic_cmp@{k}']:.2%} | {r[f'best_verified@{k}']:.2%} | "
                     f"{r[f'maj@{k}']:.2%} | {r['verifier_precision']:.2%} | {r['verifier_recall']:.2%} | {r['verifier_accuracy']:.2%} |")
    open(args.out_md, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
