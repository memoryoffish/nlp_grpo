"""
Faithful Tree-of-Thoughts self-critic for the 24-game (Yao et al., 2023).

Reproduces ToT's `value_last_step_prompt`: a FEW-SHOT completion prompt with 3 `sure`
and 3 `impossible` worked examples (the impossible examples are what break the
"yes-bias" that zero-shot critics suffer). The model judges each final candidate as
sure/impossible; we sample the judgement n_eval=3 times and aggregate with ToT's
value_map = {impossible:0.001, likely:1, sure:20}, then pick the highest-scoring
candidate. NO oracle is used to pick — the model is its own critic.

Compared head-to-head against:
  - logit-only one-word critic  (eval_tot_selfcritic.py)   [not ToT-faithful]
  - zero-shot CoT critic        (eval_selfcritic_cot.py)   [not ToT-faithful]
This script = the ToT-paper-faithful critic.

Reports: greedy, selfcritic_tot@k, best_verified@k, maj@k, and verifier precision/recall
(over all k*N judgements) so we can see whether few-shot+impossible-examples fixes yes-bias.

Usage:
  python scripts/eval_selfcritic_tot.py --model LABEL:PATH [...] \
      --test data/game24_official/test.parquet --k 16 --n_eval 3
"""
import argparse, json, os, sys, re
from collections import Counter
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verl.utils.reward_score.game24 import compute_score, extract_answer  # noqa: E402

# ---- ToT official value_last_step_prompt (princeton-nlp/tree-of-thought-llm) ----
VALUE_LAST_STEP_PROMPT = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: impossible
Input: {input}
Answer: {answer}
Judge:"""

VALUE_MAP = {"impossible": 0.001, "likely": 1.0, "sure": 20.0}
_VERDICT_RE = re.compile(r"\b(sure|likely|impossible)\b", re.I)


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


def first_verdict(text):
    m = _VERDICT_RE.search(text)
    return m.group(1).lower() if m else None


@torch.inference_mode()
def value_candidates(model, tok, numbers, exprs, n_eval, batch_size=24):
    """ToT value: for each expr, sample n_eval judgements, aggregate via value_map.
    Returns (scores[list float], votes[list dict word->count])."""
    in_str = " ".join(str(int(x)) for x in numbers)
    # one base prompt per candidate, replicated n_eval times
    flat_prompts, owner = [], []
    for j, e in enumerate(exprs):
        p = VALUE_LAST_STEP_PROMPT.format(input=in_str, answer=e if e else "(none)")
        for _ in range(n_eval):
            flat_prompts.append(p); owner.append(j)
    votes = [Counter() for _ in exprs]
    for i in range(0, len(flat_prompts), batch_size):
        chunk = flat_prompts[i:i + batch_size]; own = owner[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95,
                             max_new_tokens=8, pad_token_id=tok.eos_token_id)
        ilen = enc["input_ids"].shape[1]
        for r in range(len(chunk)):
            txt = tok.decode(out[r][ilen:], skip_special_tokens=True)
            v = first_verdict(txt)
            if v:
                votes[own[r]][v] += 1
    scores = [sum(VALUE_MAP[w] * c for w, c in vt.items()) for vt in votes]
    return scores, votes


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
    ap.add_argument("--model", action="append", required=True, help="label:path")
    ap.add_argument("--test", default="data/game24_official/test.parquet")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--out_md", default="results/selfcritic_tot.md")
    ap.add_argument("--out_json", default="results/selfcritic_tot.json")
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
        n = len(df)
        greedy_c = sc_c = bv_c = maj_c = 0
        TP = FP = FN = TN = 0
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
                scores, votes = value_candidates(model, tok, gt["numbers"],
                                                 [exprs[j] for j in cand_idx], args.n_eval)
                # confusion: pred_correct = leans 'sure' (sure votes > impossible votes)
                for jj, vt in zip(cand_idx, votes):
                    pred = vt.get("sure", 0) > vt.get("impossible", 0)
                    a = corr[jj]
                    if pred and a: TP += 1
                    elif pred and not a: FP += 1
                    elif (not pred) and a: FN += 1
                    else: TN += 1
                # ToT selection: highest value_map score (ties -> first)
                best_local = max(range(len(scores)), key=lambda t: scores[t])
                sc_c += corr[cand_idx[best_local]]
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        vacc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0.0
        res = {"n": n,
               "greedy": round(greedy_c / n, 4),
               f"selfcritic_tot@{args.k}": round(sc_c / n, 4),
               f"best_verified@{args.k}": round(bv_c / n, 4),
               f"maj@{args.k}": round(maj_c / n, 4),
               "verifier_precision": round(prec, 4),
               "verifier_recall": round(rec, 4),
               "verifier_accuracy": round(vacc, 4),
               "confusion": {"TP": TP, "FP": FP, "FN": FN, "TN": TN}}
        table[label] = res
        print(json.dumps(res, indent=2), flush=True)
        del model; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(table, open(args.out_json, "w"), indent=2)
    k = args.k
    lines = [f"ToT-faithful self-critic (value_last_step_prompt, few-shot, n_eval={args.n_eval}, k={k}); test={args.test}",
             "selfcritic_tot@k = model self-judges via ToT few-shot value prompt, picks best (no oracle).",
             "", f"| Model | greedy | selfcritic_tot@{k} | best_verified@{k} | maj@{k} | verif_prec | verif_rec | verif_acc |",
             "|---|---|---|---|---|---|---|---|"]
    for lb, r in table.items():
        lines.append(f"| {lb} | {r['greedy']:.2%} | {r[f'selfcritic_tot@{k}']:.2%} | "
                     f"{r[f'best_verified@{k}']:.2%} | {r[f'maj@{k}']:.2%} | "
                     f"{r['verifier_precision']:.2%} | {r['verifier_recall']:.2%} | {r['verifier_accuracy']:.2%} |")
    open(args.out_md, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
