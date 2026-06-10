#!/bin/bash
# GRPO training for the 24-point game on the uv (py3.11) env using the HF rollout backend.
#
# Why HF rollout: vLLM in this env is ABI-broken (built for CUDA 12.x vs torch cu118) and
# there is no conda `zero` env, so the vLLM path in scripts/train_game24_local.sh does not
# run here. This script mirrors that config but uses actor_rollout_ref.rollout.name=hf.
#
# This is the workhorse for the SFT->GRPO vs pure-GRPO comparison: the only thing that
# changes between the two arms is BASE_MODEL (base model vs SFT checkpoint).
#
#   # Arm A — pure GRPO from the base model
#   EXPERIMENT_NAME=game24-grpo-base bash scripts/train_game24_grpo_hf.sh
#
#   # Arm B — GRPO from the SFT checkpoint
#   BASE_MODEL=$(ls -td checkpoints/TinyZero/game24-sft/global_step_* | head -1) \
#   EXPERIMENT_NAME=game24-grpo-sftinit bash scripts/train_game24_grpo_hf.sh

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

VENV_DIR=${VENV_DIR:-"/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311"}
PYTHON=${PYTHON:-"$VENV_DIR/bin/python"}
RAY=${RAY:-"$VENV_DIR/bin/ray"}

BASE_MODEL=${BASE_MODEL:-"/mnt/workspace/akide/models/Qwen2.5-1.5B-Instruct"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/data/game24"}
VAL_FILES=${VAL_FILES:-"$DATA_DIR/test.parquet"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"game24-grpo-hf"}
LOG_FILE=${LOG_FILE:-"$PROJECT_ROOT/${EXPERIMENT_NAME}.log"}

# Scale (kept modest because HF rollout generation is much slower than vLLM).
# NOTE: actual steps = min(TOTAL_STEPS, TOTAL_EPOCHS * ceil(train_rows/TRAIN_BATCH_SIZE)).
# With 1262 rows / batch 16 = 78 steps/epoch, so TOTAL_EPOCHS must be >=3 to reach 160 steps.
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-200}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-50}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-320}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-300}
LR=${LR:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
KL_COEF=${KL_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
REF_LOG_PROB_MICRO_BATCH_SIZE=${REF_LOG_PROB_MICRO_BATCH_SIZE:-4}
N_GPUS=${N_GPUS:-1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
FINAL_VAL=${FINAL_VAL:-True}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console']"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/mnt/workspace/akide/models/huggingface}
export WANDB_MODE=${WANDB_MODE:-offline}
export VERL_DISABLE_FLASH_ATTN_CE=${VERL_DISABLE_FLASH_ATTN_CE:-1}

if [[ ! -d "$BASE_MODEL" ]]; then echo "Missing model dir: $BASE_MODEL" >&2; exit 1; fi
if [[ ! -f "$DATA_DIR/train.parquet" ]]; then echo "Missing $DATA_DIR/train.parquet (run prepare_data_game24.sh)" >&2; exit 1; fi

# SKIP_RAY_STOP=1 lets a second arm run concurrently on its own isolated Ray
# cluster (set RAY_ADDRESS) without killing a first arm's cluster.
if [[ "${SKIP_RAY_STOP:-0}" != "1" ]]; then
    "$RAY" stop --force >/dev/null 2>&1 || true
fi

"$PYTHON" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$TOP_P" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef="$KL_COEF" \
    trainer.critic_warmup=0 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name=TinyZero \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    +trainer.final_val="$FINAL_VAL" \
    trainer.default_hdfs_dir=null \
    2>&1 | tee "$LOG_FILE"
