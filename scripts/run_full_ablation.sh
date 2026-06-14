#!/usr/bin/env bash
# Hands-off chain: train SFT-single (single-solution SFT), then run the 8-arm GRPO ablation suite.
# Launch ONLY after SFT-multi (game24-sft-official) has finished and freed GPUs 1-4.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"

echo "===== [1/2] Train SFT-single ($(date '+%F %T')) ====="
env CUDA_VISIBLE_DEVICES=1,2,3,4 N_GPUS=4 \
  SFT_DATA_DIR="$ROOT/data/sft_game24_single" \
  SFT_SAVE_DIR="$ROOT/checkpoints/TinyZero/game24-sft-single" \
  LR=1e-5 TOTAL_EPOCHS=4 EXPERIMENT_NAME=game24-sft-single \
  LOG_FILE="$ROOT/game24_sft_single.log" \
  bash scripts/train_sft_game24.sh

echo "===== [2/2] Run 8-arm GRPO ablation suite ($(date '+%F %T')) ====="
bash scripts/run_ablation_suite.sh
echo "===== FULL ABLATION CHAIN DONE ($(date '+%F %T')) ====="
