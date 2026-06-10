"""
OOD evaluation: load a trained checkpoint and score it on any parquet test set.

Supports both game24 and countdown reward functions, dispatched automatically
via the `data_source` field in the parquet (same logic as the trainer).

Usage:
    python examples/eval_ood.py \
        --model  checkpoints/TinyZero/game24-qwen2.5-1.5b-grpo-local/actor/global_step_200 \
        --data   data/countdown/test.parquet \
        --output results/ood_countdown_step200.json

    # Evaluate on multiple files at once:
    python examples/eval_ood.py \
        --model  checkpoints/TinyZero/.../actor/global_step_200 \
        --data   data/game24/test.parquet data/countdown/test.parquet \
        --output results/ood_eval.json
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
from verl.utils.reward_score import game24, countdown


def _select_score_fn(data_source: str):
    if "countdown" in data_source:
        return countdown.compute_score
    if "game24" in data_source or "24-game" in data_source:
        return game24.compute_score
    raise ValueError(f"Unknown data_source: {data_source!r}")


def build_prompt(row) -> str:
    """Extract the raw prompt string from a parquet row."""
    prompt_field = row["prompt"]
    if isinstance(prompt_field, list):
        # list of chat dicts — use last user message content
        for msg in reversed(prompt_field):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg["content"]
        return prompt_field[-1]["content"] if prompt_field else ""
    return str(prompt_field)


@torch.inference_mode()
def evaluate(model_path: str, parquet_files: list[str], batch_size: int,
             max_new_tokens: int, device: str) -> dict:
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    all_results = {}

    for parquet_file in parquet_files:
        print(f"\nEvaluating: {parquet_file}")
        df = pd.read_parquet(parquet_file)

        scores = []
        for i in tqdm(range(0, len(df), batch_size), desc="Batches"):
            batch = df.iloc[i: i + batch_size]

            prompts = [build_prompt(row) for _, row in batch.iterrows()]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for j, (_, row) in enumerate(batch.iterrows()):
                input_len = inputs["input_ids"].shape[1]
                generated = tokenizer.decode(
                    outputs[j][input_len:], skip_special_tokens=False
                )
                full_str = prompts[j] + generated

                data_source = str(row.get("data_source", "game24"))
                ground_truth = row["reward_model"]["ground_truth"] \
                    if isinstance(row["reward_model"], dict) \
                    else row["reward_model"].get("ground_truth", {})

                score_fn = _select_score_fn(data_source)
                score = score_fn(solution_str=full_str, ground_truth=ground_truth)
                scores.append(score)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        correct = sum(1 for s in scores if s >= 1.0)
        label = os.path.splitext(os.path.basename(parquet_file))[0]
        data_source = str(df.iloc[0].get("data_source", label))

        all_results[data_source] = {
            "file": parquet_file,
            "n_samples": len(scores),
            "mean_score": round(mean_score, 4),
            "accuracy": round(correct / len(scores), 4) if scores else 0.0,
            "n_correct": correct,
        }
        print(f"  {data_source}: mean={mean_score:.4f}  accuracy={correct}/{len(scores)}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="OOD evaluation on any parquet test set")
    parser.add_argument("--model", required=True,
                        help="Path to trained model checkpoint directory")
    parser.add_argument("--data", nargs="+", required=True,
                        help="One or more parquet test files")
    parser.add_argument("--output", default=None,
                        help="Save results JSON to this path (optional)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results = evaluate(
        model_path=args.model,
        parquet_files=args.data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    print("\n===== Summary =====")
    for src, r in results.items():
        print(f"  {src}: accuracy={r['accuracy']:.2%}  ({r['n_correct']}/{r['n_samples']})")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
