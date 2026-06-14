"""
Tree-of-Thoughts-aligned best-of-N evaluation: the model is its OWN critic.

For each puzzle:
  1. sample k candidate solutions (temperature sampling, R1 prompt);
  2. the SAME model self-evaluates each candidate (ToT-style value: sure/likely/impossible),
     scored from its own logits — NO ground-truth verifier is used to pick;
  3. select the candidate with the highest self-critic score (ties -> first);
  4. report whether the model-selected candidate is actually correct (rule verifier, for scoring only).

Reported per model:
  greedy            : 1-shot deterministic accuracy
  selfcritic@k      : ToT-aligned, deployable WITHOUT an oracle (model picks)  <-- the headline ToT number
  best_verified@k   : an oracle picks any correct sample (upper bound)
  maj@k             : majority-vote expression (no critic)
The gap selfcritic@k vs best_verified@k = how good the model is at judging its own answers.

Usage:
  python scripts/eval_tot_selfcritic.py --model LABEL:PATH [--model ...] \
      --test data/game24_official/test.parquet --k 16 --temperature 1.0 --out_md results/tot_selfcritic.md
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


CRITIC_SYS = "You are a strict evaluator for the 24-game."


def critic_prompt(tok, numbers, expr):
    user = (
        f"Numbers: {list(numbers)}. Candidate expression: {expr}\n"
        f"Decide whether this expression uses each of the numbers exactly once AND evaluates "
        f"to exactly 24. Reply with ONE word only: sure (definitely correct), likely (maybe), "
        f"or impossible (definitely wrong)."
    )
    msgs = [{"role": "system", "content": CRITIC_SYS}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.inference_mode()
def critic_scores(model, tok, numbers, exprs, batch_size=16):
    """Return a self-critic score in [0,2] for each candidate expr, from the model's own logits
    over {sure, likely, impossible}. Higher = the model is more confident it is correct."""
    # token ids for the three verdict words (first sub-token, with and without leading space)
    def tid(w):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        return ids[0] if ids else None
    sure_ids = {tid("sure"), tid(" sure"), tid("Sure"), tid(" Sure")} - {None}
    likely_ids = {tid("likely"), tid(" likely"), tid("Likely"), tid(" Likely")} - {None}
    imposs_ids = {tid("impossible"), tid(" impossible"), tid("Impossible"), tid(" Impossible")} - {None}
    scores = []
    for i in range(0, len(exprs), batch_size):
        chunk = exprs[i:i + batch_size]
        prompts = [critic_prompt(tok, numbers, e) for e in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits  # (b, T, V)
        last = logits[:, -1, :]       # next-token logits
        lp = torch.log_softmax(last.float(), dim=-1)
        for r in range(len(chunk)):
            p_sure = torch.logsumexp(lp[r, list(sure_ids)], 0) if sure_ids else torch.tensor(-1e9)
            p_like = torch.logsumexp(lp[r, list(likely_ids)], 0) if likely_ids else torch.tensor(-1e9)
            p_imp = torch.logsumexp(lp[r, list(imposs_ids)], 0) if imposs_ids else torch.tensor(-1e9)
            # expected value: sure=2, likely=1, impossible=0 (softmax over the 3)
            v = torch.softmax(torch.stack([p_imp, p_like, p_sure]), 0)
            scores.append(float(v[1] * 1 + v[2] * 2))
    return scores


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
    ap.add_argument("--out_md", default="results/tot_selfcritic.md")
    ap.add_argument("--out_json", default="results/tot_selfcritic.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.test)
    table = {}
    for spec in args.model:
        label, _, path = spec.partition(":")
        print(f"\n===== {label} ({path}) =====")
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).cuda().eval()
        n = len(df); greedy_c = sc_c = bv_c = maj_c = 0
        for _, row in tqdm(df.iterrows(), total=n, desc=label):
            prompt = build_prompt(row); gt = gt_of(row)
            # greedy (1-shot deterministic)
            penc = tok(prompt, return_tensors="pt").to(model.device)
            gout = model.generate(**penc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                  pad_token_id=tok.eos_token_id)
            gtxt = tok.decode(gout[0][penc["input_ids"].shape[1]:], skip_special_tokens=False)
            greedy_c += compute_score(prompt + gtxt, gt) >= 1.0
            # k samples
            samples = sample_candidates(model, tok, prompt, args.k, args.temperature, args.max_new_tokens)
            exprs = [extract_answer(prompt + s) or "" for s in samples]
            corr = [compute_score(prompt + s, gt) >= 1.0 for s in samples]
            # best_verified@k (oracle upper bound)
            bv_c += any(corr)
            # maj@k
            valid = [e for e in exprs if e]
            if valid:
                top = Counter(valid).most_common(1)[0][0]
                maj_c += compute_score(prompt + f"<answer>{top}</answer>", gt) >= 1.0
            # selfcritic@k (ToT: model picks)
            cand_idx = [j for j, e in enumerate(exprs) if e]
            if cand_idx:
                cs = critic_scores(model, tok, gt["numbers"], [exprs[j] for j in cand_idx])
                best = cand_idx[int(max(range(len(cs)), key=lambda t: cs[t]))]
                sc_c += corr[best]
        res = {"n": n,
               "greedy": round(greedy_c / n, 4),
               f"selfcritic@{args.k}": round(sc_c / n, 4),
               f"best_verified@{args.k}": round(bv_c / n, 4),
               f"maj@{args.k}": round(maj_c / n, 4)}
        table[label] = res
        print(json.dumps(res, indent=2))
        del model; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(table, open(args.out_json, "w"), indent=2)
    k = args.k
    lines = [f"ToT-aligned self-critic best-of-N (k={k}, T={args.temperature}); test={args.test}",
             "selfcritic@k = model picks among k (no oracle, ToT); best_verified@k = oracle upper bound; maj@k = majority vote.",
             "", f"| Model | greedy | selfcritic@{k} | best_verified@{k} | maj@{k} |", "|---|---|---|---|---|"]
    for lb, r in table.items():
        lines.append(f"| {lb} | {r['greedy']:.2%} | {r[f'selfcritic@{k}']:.2%} | {r[f'best_verified@{k}']:.2%} | {r[f'maj@{k}']:.2%} |")
    open(args.out_md, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
