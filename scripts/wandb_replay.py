"""
Replay per-step metrics from verl console logs into wandb runs.

Our GRPO runs used trainer.logger=['console'] (wandb.ai was unreliable for live
sync), so the real experiments never reached wandb. This re-creates proper wandb
runs from the console logs (same parsing as scripts/analyze_grpo.py), tagged with
each run's config, so the dashboard reflects the actual recent experiments.

Usage: python scripts/wandb_replay.py            # online (uses ~/.netrc login)
       WANDB_MODE=offline python scripts/wandb_replay.py
"""
import os, re, glob
import wandb

STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
KV_RE = re.compile(r"([\w/.]+):(-?[\d.eE+]+)")

PROJECT = os.environ.get("WANDB_PROJECT", "TinyZero")
# logfile -> (display name, config)
RUNS = {
    "game24-grpo-base.log":        ("game24-grpo-base (pure-GRPO)",      dict(init="base", reward="sparse", n=4, lr=1e-6, kl=0.001, ent=0.001, temp=1.0)),
    "game24-grpo-sftinit.log":     ("game24-grpo-sftinit (SFT->GRPO v1)", dict(init="sft-v1", reward="sparse", n=4, lr=1e-6, kl=0.001, ent=0.001, temp=1.0)),
    "game24-grpo-v2-control.log":  ("game24-grpo-v2-control",            dict(init="sft-v2", reward="sparse", n=4, lr=1e-6, kl=0.001, ent=0.001, temp=1.0)),
    "game24-grpo-v2-n8.log":       ("game24-grpo-v2-n8",                 dict(init="sft-v2", reward="sparse", n=8, lr=1e-6, kl=0.001, ent=0.001, temp=1.0)),
    "game24-grpo-v2-shaped.log":   ("game24-grpo-v2-shaped",             dict(init="sft-v2", reward="shaped", n=4, lr=1e-6, kl=0.001, ent=0.001, temp=1.0)),
    "game24-grpo-v2-staged-p1.log":("game24-grpo-v2-staged-p1",          dict(init="sft-v2", reward="shaped->sparse", n=4, lr=1e-6, kl=0.001, ent=0.001, temp=1.0)),
    "game24-grpo-v2-divA.log":     ("game24-grpo-v2-divA",               dict(init="sft-v2", reward="shaped", n=4, lr=1e-6, kl=0.005, ent=0.01, temp=1.2)),
    "game24-grpo-v2-divB.log":     ("game24-grpo-v2-divB",               dict(init="sft-v2", reward="shaped", n=8, lr=1e-6, kl=0.01,  ent=0.02, temp=1.2)),
    "game24-grpo-v2-divC.log":     ("game24-grpo-v2-divC (hi-KL)",       dict(init="sft-v2", reward="shaped", n=4, lr=1e-6, kl=0.03,  ent=0.01, temp=1.2)),
    "game24-grpo-v2-divD.log":     ("game24-grpo-v2-divD (hi-ent)",      dict(init="sft-v2", reward="shaped", n=4, lr=1e-6, kl=0.005, ent=0.03, temp=1.3)),
}


def parse(path):
    rows = {}
    for line in open(path, errors="ignore"):
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        kv = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
        rows.setdefault(step, {}).update(kv)
    return rows


def main():
    os.environ.setdefault("WANDB_SILENT", "true")
    for logf, (name, cfg) in RUNS.items():
        if not os.path.exists(logf):
            print(f"skip (missing): {logf}")
            continue
        rows = parse(logf)
        if not rows:
            print(f"skip (no steps): {logf}")
            continue
        run = wandb.init(project=PROJECT, name=name, config=cfg, reinit=True,
                         tags=["game24", "replay", cfg["reward"]])
        for step in sorted(rows):
            wandb.log(rows[step], step=step)
        run.finish()
        print(f"logged {len(rows):3d} steps -> wandb run '{name}'")


if __name__ == "__main__":
    main()
