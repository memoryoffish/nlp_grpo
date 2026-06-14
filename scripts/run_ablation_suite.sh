#!/usr/bin/env bash
# Compliant ablation suite: 8 GRPO arms, SEQUENTIAL, 4 GPUs each (GPUs 1-4), on the
# assignment-correct data (train=nlile-minus-test, test=ttc idx900-999). One factor varies per arm.
# Reuses scripts/train_game24_grpo_hf.sh (parameterized) + GAME24_REWARD env gate for shaped.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
DATA="$ROOT/data/game24_official"
BASE_MODEL_PRETRAINED="/mnt/workspace/akide/models/Qwen2.5-1.5B-Instruct"
SFT_MULTI=$(ls -td "$ROOT"/checkpoints/TinyZero/game24-sft-official/global_step_* 2>/dev/null | head -1)
SFT_SINGLE=$(ls -td "$ROOT"/checkpoints/TinyZero/game24-sft-single/global_step_* 2>/dev/null | head -1)
echo "SFT_MULTI=$SFT_MULTI"; echo "SFT_SINGLE=$SFT_SINGLE"
[ -d "$SFT_MULTI" ] || { echo "missing SFT_MULTI"; exit 1; }
[ -d "$SFT_SINGLE" ] || { echo "missing SFT_SINGLE"; exit 1; }

run () {  # $1=name $2=base $3=n $4=reward $5=lr
  local name=$1 base=$2 n=$3 rew=$4 lr=$5
  echo "================== $name | base=$(basename $(dirname $base 2>/dev/null) 2>/dev/null)/$(basename $base) n=$n rew=$rew lr=$lr | $(date '+%F %T') =================="
  local REW_ENV=""; [ "$rew" = "shaped" ] && REW_ENV="GAME24_REWARD=shaped"
  env CUDA_VISIBLE_DEVICES=1,2,3,4 N_GPUS=4 DATA_DIR="$DATA" \
    TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 PPO_MICRO_BATCH_SIZE=4 REF_LOG_PROB_MICRO_BATCH_SIZE=4 \
    GRAD_CKPT=True MAX_RESPONSE_LENGTH=320 MAX_PROMPT_LENGTH=256 \
    ROLLOUT_N=$n LR=$lr TEMPERATURE=1.0 KL_COEF=0.001 ENTROPY_COEFF=0.001 \
    TOTAL_EPOCHS=1 TOTAL_STEPS=200 TEST_FREQ=40 SAVE_FREQ=40 VAL_BEFORE_TRAIN=True FINAL_VAL=True \
    $REW_ENV BASE_MODEL="$base" EXPERIMENT_NAME="$name" \
    bash scripts/train_game24_grpo_hf.sh || echo "!!! $name FAILED (continuing) !!!"
  echo "================== $name DONE | $(date '+%F %T') =================="
}

#    name                        base           n  reward  lr      # ablation axis
run abl-C-multi-n4-sparse-lr1e6  "$SFT_MULTI"   4  sparse  1e-6   # control
run abl-A1-n8                    "$SFT_MULTI"   8  sparse  1e-6   # rollout_n
run abl-A2-single                "$SFT_SINGLE"  4  sparse  1e-6   # 多解 (single-sol SFT)
run abl-A3-shaped                "$SFT_MULTI"   4  shaped  1e-6   # reward
run abl-A4a-lr5e7                "$SFT_MULTI"   4  sparse  5e-7   # lr
run abl-A4b-lr2e6                "$SFT_MULTI"   4  sparse  2e-6   # lr
run abl-A4c-lr5e6                "$SFT_MULTI"   4  sparse  5e-6   # lr
run abl-R1-puregrpo              "$BASE_MODEL_PRETRAINED" 4 sparse 1e-6  # SFT 有无 (no SFT)
echo "===== ABLATION SUITE DONE ($(date '+%F %T')) ====="
