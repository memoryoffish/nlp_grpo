#!/bin/bash
# SFT cold-start for Game-of-24.
# Trains on programmatically generated ToT reasoning traces so the model
# learns to explore combinations before giving an answer.  Run this BEFORE
# GRPO to give the policy a useful starting point.
#
# Full two-stage workflow:
#   1. bash scripts/prepare_data_game24.sh          # build GRPO data
#   2. bash scripts/generate_sft_data_game24.sh     # build SFT traces
#   3. bash scripts/train_sft_game24.sh             # SFT warm-start  ← this script
#   4. BASE_MODEL=<sft-ckpt> bash scripts/train_game24_local.sh   # GRPO
#
# Usage (from project root) — this machine uses a uv venv (py3.11), not conda `zero`:
#   bash scripts/train_sft_game24.sh

set -x

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

VENV_DIR=${VENV_DIR:-"/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311"}
TORCHRUN=${TORCHRUN:-"$VENV_DIR/bin/torchrun"}

BASE_MODEL=${BASE_MODEL:-"/mnt/workspace/akide/models/Qwen2.5-1.5B-Instruct"}
SFT_DATA_DIR=${SFT_DATA_DIR:-"$PROJECT_ROOT/data/sft_game24"}
SFT_SAVE_DIR=${SFT_SAVE_DIR:-"$PROJECT_ROOT/checkpoints/TinyZero/game24-sft"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"game24-sft"}
LOG_FILE=${LOG_FILE:-"$PROJECT_ROOT/game24_sft.log"}

N_GPUS=${N_GPUS:-1}
MAX_LENGTH=${MAX_LENGTH:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
LR=${LR:-2e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console']"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/mnt/workspace/akide/models/huggingface}
export WANDB_MODE=${WANDB_MODE:-offline}
export VERL_DISABLE_FLASH_ATTN_CE=${VERL_DISABLE_FLASH_ATTN_CE:-1}

"$TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$N_GPUS" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$SFT_DATA_DIR/train.parquet" \
    data.val_files="$SFT_DATA_DIR/val.parquet" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length="$MAX_LENGTH" \
    data.micro_batch_size="$MICRO_BATCH_SIZE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    model.partial_pretrain="$BASE_MODEL" \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.cpu_offload=False \
    optim.lr="$LR" \
    optim.clip_grad=1.0 \
    optim.warmup_steps_ratio=0.05 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_local_dir="$SFT_SAVE_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.project_name=TinyZero \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" 2>&1 | tee "$LOG_FILE"

echo ""
echo "SFT checkpoint saved to: $SFT_SAVE_DIR"
echo "To continue with GRPO, run:"
echo "  BASE_MODEL=\$(ls -td $SFT_SAVE_DIR/global_step_* | head -1) \\"
echo "  bash scripts/train_game24_local.sh"
