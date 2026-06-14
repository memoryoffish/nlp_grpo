"""
Generative (chain-of-thought) self-critic — the FAIR version of model-as-critic.

Motivation: the logit-only ToT value prompt (eval_tot_selfcritic.py) forces a one-word
verdict in a single forward pass, giving the model NO chance to actually COMPUTE the
expression. For 24-game, verifying "(8-4)*(6-0)==24" requires arithmetic. Here the model
is allowed to reason step by step, then emit FINAL: CORRECT / WRONG. We measure:

  greedy              : 1-shot accuracy (reference)
  selfcritic_cot@k    : model reasons, picks first candidate it marks CORRECT (deployable, no oracle)
  best_verified@k     : oracle rule-verifier upper bound
  maj@k               : majority vote
  + verifier precision / recall / accuracy over ALL k*N candidate-judgements
    (TP = model says CORRECT and it really is; this directly answers
     "is the model a good verifier when allowed to compute?")

Usage:
  python scripts/eval_selfcritic_cot.py --model LABEL:PATH [--model ...] \
      --test data/game24_official/test.parquet --k 16 --out_md results/selfcritic_cot.md
"""
import argparse, json, os, sys, re
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


CRITIC_SYS = "You are a meticulous checker for the 24-game. You verify by computing, step by step."


def critic_prompt(tok, numbers, expr):
    user = (
        f"Numbers: {list(numbers)}\n"
        f"Candidate expression: {expr}\n\n"
        f"Verify two things:\n"
        f"  (1) the expression uses each of the four numbers exactly once;\n"
        f"  (2) it evaluates to exactly 24.\n"
        f"Compute the value step by step (show the arithmetic), then on the LAST line output "
        f"exactly one of:\nFINAL: CORRECT\nFINAL: WRONG"
    )
    msgs = [{"role": "system", "content": CRITIC_SYS}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


_VERDICT_RE = re.compile(r"FINAL:\s*\**\s*(CORRECT|WRONG)", re.I)
_FALLBACK_RE = re.compile(r"\b(CORRECT|WRONG)\b", re.I)


def parse_verdict(text):
    m = _VERDICT_RE.findall(text)
    if not m:
        m = _FALLBACK_RE.findall(text)
    if not m:
        return None
    return m[-1].upper() == "CORRECT"


@torch.inference_mode()
def critic_judge(model, tok, numbers, exprs, max_new_tokens=220, batch_size=16):
    """Return list of bool|None: model's CoT verdict (True=CORRECT) for each candidate expr."""
    verdicts = []
    for i in range(0, len(exprs), batch_size):
        chunk = exprs[i:i + batch_size]
        prompts = [critic_prompt(tok, numbers, e) for e in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id)
        ilen = enc["input_ids"].shape[1]
        for r in range(len(chunk)):
            txt = tok.decode(out[r][ilen:], skip_special_tokens=True)
            verdicts.append(parse_verdict(txt))
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
    ap.add_argument("--model", action="append", required=True, help="label:path")
    ap.add_argument("--test", default="data/game24_official/test.parquet")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--critic_max_new_tokens", type=int, default=220)
    ap.add_argument("--out_md", default="results/selfcritic_cot.md")
    ap.add_argument("--out_json", default="results/selfcritic_cot.json")
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
                verdicts = critic_judge(model, tok, gt["numbers"], [exprs[j] for j in cand_idx],
                                        max_new_tokens=args.critic_max_new_tokens)
                # verifier confusion over all judged candidates
                for jj, v in zip(cand_idx, verdicts):
                    if v is None:
                        continue
                    a = corr[jj]
                    if v and a: TP += 1
                    elif v and not a: FP += 1
                    elif (not v) and a: FN += 1
                    else: TN += 1
                # deploy: pick FIRST candidate the critic marks CORRECT; else fall back to candidate[0]
                pick = next((jj for jj, v in zip(cand_idx, verdicts) if v), cand_idx[0])
                sc_c += corr[pick]
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        vacc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0.0
        res = {"n": n,
               "greedy": round(greedy_c / n, 4),
               f"selfcritic_cot@{args.k}": round(sc_c / n, 4),
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
    lines = [f"Generative CoT self-critic (k={k}, T={args.temperature}); test={args.test}",
             "selfcritic_cot@k = model reasons step-by-step, picks first candidate it marks CORRECT (no oracle).",
             "verifier_* = how good the model is at JUDGING a candidate when allowed to compute.",
             "", f"| Model | greedy | selfcritic_cot@{k} | best_verified@{k} | maj@{k} | verif_prec | verif_rec | verif_acc |",
             "|---|---|---|---|---|---|---|---|"]
    for lb, r in table.items():
        lines.append(f"| {lb} | {r['greedy']:.2%} | {r[f'selfcritic_cot@{k}']:.2%} | "
                     f"{r[f'best_verified@{k}']:.2%} | {r[f'maj@{k}']:.2%} | "
                     f"{r['verifier_precision']:.2%} | {r['verifier_recall']:.2%} | {r['verifier_accuracy']:.2%} |")
    open(args.out_md, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
