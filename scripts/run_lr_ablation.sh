#!/usr/bin/env bash
# Learning-rate ablation for GRPO (only LR varies; everything else aligned with the v2 arms).
# Base = SFT v2 (global_step_296); sparse reward; n=4; KL=0.001; ent=0.001; T=1.0; max_response=320;
# 1 epoch (~78 steps), val at 0/40/final.
#
# SEQUENTIAL, 4 GPUs per arm (GPUs 1-4): one Ray cluster at a time -> no port conflicts, and 4-GPU
# data-parallel rollout is fast (single-GPU HF rollout stalled on validation; two concurrent Ray
# heads collided on Ray's fixed aux ports e.g. 10002). Each arm ~1h; 4 arms ~4h total.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
SFT=${SFT:-"$ROOT/checkpoints/TinyZero/game24-sft-v2/global_step_296"}

for lr in 5e-7 1e-6 2e-6 5e-6; do
  echo "================= LR=$lr  ($(date '+%T')) ================="
  CUDA_VISIBLE_DEVICES=1,2,3,4 N_GPUS=4 \
    TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 PPO_MICRO_BATCH_SIZE=4 REF_LOG_PROB_MICRO_BATCH_SIZE=4 \
    GRAD_CKPT=True \
    ROLLOUT_N=4 MAX_RESPONSE_LENGTH=320 MAX_PROMPT_LENGTH=256 LR=$lr TEMPERATURE=1.0 \
    KL_COEF=0.001 ENTROPY_COEFF=0.001 \
    TOTAL_EPOCHS=1 TOTAL_STEPS=200 TEST_FREQ=40 SAVE_FREQ=40 VAL_BEFORE_TRAIN=True FINAL_VAL=True \
    BASE_MODEL="$SFT" EXPERIMENT_NAME=game24-grpo-v2-lr$lr \
    bash scripts/train_game24_grpo_hf.sh
  echo "================= LR=$lr DONE ($(date '+%T')) ================="
done
echo "===== ALL LR ARMS DONE ====="
