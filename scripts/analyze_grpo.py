"""
Parse one or more verl GRPO console logs into per-step metric CSVs and a
comparison plot. Used to compare the pure-GRPO arm against the SFT->GRPO arm.

Usage:
  python scripts/analyze_grpo.py \
      --log game24-grpo-base.log:pure-GRPO \
      --log game24-grpo-sftinit.log:SFT->GRPO \
      --out_csv results/grpo_curves.csv \
      --out_png results/grpo_curves.png
"""
import argparse
import re
import csv
import os

STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
KV_RE = re.compile(r"([\w/.]+):(-?[\d.eE+]+)")

# metrics we care about for the report
KEYS = [
    "critic/score/mean",
    "critic/score/max",
    "val/test_score/game24",
    "response_length/mean",
    "actor/kl_loss",
    "actor/entropy_loss",
]


def parse_log(path):
    """Return {step: {metric: value}} merged across lines (val lines and train
    lines for the same step are merged)."""
    rows = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            kvs = dict((k, float(v)) for k, v in KV_RE.findall(m.group(2)))
            rows.setdefault(step, {}).update(kvs)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True,
                    help="path:label (label optional)")
    ap.add_argument("--out_csv", default="results/grpo_curves.csv")
    ap.add_argument("--out_png", default="results/grpo_curves.png")
    args = ap.parse_args()

    series = {}
    for spec in args.log:
        path, _, label = spec.partition(":")
        label = label or os.path.basename(path)
        series[label] = parse_log(path)
        n = len(series[label])
        print(f"{label}: parsed {n} steps from {path}")

    # write long-format CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step"] + KEYS)
        for label, rows in series.items():
            for step in sorted(rows):
                w.writerow([label, step] + [rows[step].get(k, "") for k in KEYS])
    print(f"wrote {args.out_csv}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = ["critic/score/mean", "val/test_score/game24",
                  "response_length/mean", "actor/kl_loss"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, key in zip(axes.flat, panels):
            for label, rows in series.items():
                xs = [s for s in sorted(rows) if key in rows[s]]
                ys = [rows[s][key] for s in xs]
                if xs:
                    ax.plot(xs, ys, marker="o", ms=3, label=label)
            ax.set_title(key)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=120)
        print(f"wrote {args.out_png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
