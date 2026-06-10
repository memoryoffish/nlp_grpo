#!/bin/bash
# Smoke test for the SFT cold-start pipeline.
# Self-contained: generates a tiny synthetic dataset from hardcoded puzzles
# (no prior data preparation needed), then runs 1 SFT training epoch.
# Expected runtime: 2–4 minutes on a single GPU.
#
# Usage (from project root):
#   conda activate zero
#   bash scripts/smoke_test_sft_game24.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BASE_MODEL=${BASE_MODEL:-"$PWD/models/Qwen2.5-1.5B-Instruct"}
SMOKE_DIR="$PWD/data/sft_game24_smoke"
mkdir -p "$SMOKE_DIR"

echo "=========================================="
echo "STEP 1: Generate tiny SFT dataset"
echo "=========================================="

# Write the data-gen script to a real file to avoid conda run + heredoc stdin issues
GEN_SCRIPT="$SMOKE_DIR/gen_smoke_data.py"
cat > "$GEN_SCRIPT" << 'PYEOF'
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from examples.data_preprocess.generate_sft_game24 import (
    solve24, generate_trace, make_question
)
from datasets import Dataset

SMOKE_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(0)

PUZZLES = [
    [1, 2, 3, 4], [2, 3, 4, 6], [1, 3, 4, 6], [2, 4, 6, 8], [1, 4, 6, 8],
    [3, 3, 8, 8], [1, 5, 5, 5], [2, 5, 6, 8], [3, 4, 6, 8], [2, 3, 6, 8],
    [1, 2, 6, 8], [4, 4, 6, 8], [1, 6, 6, 8], [2, 6, 6, 8], [4, 6, 6, 8],
    [1, 2, 4, 8], [2, 4, 4, 8], [3, 3, 6, 8], [2, 2, 6, 8], [1, 1, 4, 6],
]

samples = []
for nums in PUZZLES:
    trace = generate_trace(nums)
    if trace is not None:
        samples.append({'prompt': make_question(nums), 'response': trace})

assert len(samples) >= 8, f"Too few solvable puzzles: {len(samples)}"
random.shuffle(samples)
print(f"Generated {len(samples)} samples")

val_samples   = samples[:4]
train_samples = samples[4:]
Dataset.from_list(train_samples).to_parquet(f'{SMOKE_DIR}/train.parquet')
Dataset.from_list(val_samples).to_parquet(f'{SMOKE_DIR}/val.parquet')
print(f"Saved train({len(train_samples)}) + val({len(val_samples)}) -> {SMOKE_DIR}/")

ex = train_samples[0]
print("\n--- Sample prompt ---")
print(ex['prompt'])
print("\n--- Sample response ---")
print(ex['response'])
print("\nData generation OK")
PYEOF

conda run -n zero python3 "$GEN_SCRIPT"

echo ""
echo "=========================================="
echo "STEP 2: SFT training smoke (1 epoch)"
echo "=========================================="

conda run -n zero python3 -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$SMOKE_DIR/train.parquet" \
    data.val_files="$SMOKE_DIR/val.parquet" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=512 \
    data.micro_batch_size=2 \
    data.train_batch_size=8 \
    model.partial_pretrain="$BASE_MODEL" \
    model.enable_gradient_checkpointing=False \
    model.fsdp_config.cpu_offload=False \
    optim.lr=2e-5 \
    optim.clip_grad=1.0 \
    optim.warmup_steps_ratio=0.0 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$SMOKE_DIR/checkpoints" \
    trainer.default_hdfs_dir=null \
    trainer.project_name=TinyZero \
    trainer.experiment_name=sft-smoke-test \
    trainer.logger=['console']

echo ""
echo "=========================================="
echo "SFT smoke test PASSED — pipeline is working."
echo "Next: generate full SFT data and run training:"
echo "  bash scripts/generate_sft_data_game24.sh"
echo "  bash scripts/train_sft_game24.sh"
echo "=========================================="
