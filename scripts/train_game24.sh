#!/bin/bash
# GRPO training for 24-point game with Qwen2.5-1.5B-Instruct
#
# Usage:
#   export BASE_MODEL=/path/to/Qwen2.5-1.5B-Instruct   # or HF model id
#   export DATA_DIR=~/data/game24
#   export N_GPUS=1                                      # 1 for single GPU
#   bash scripts/train_game24.sh
#
# For multi-GPU (e.g. 2 GPUs):
#   export N_GPUS=2
#   export ROLLOUT_TP_SIZE=2

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

N_GPUS=${N_GPUS:-1}
BASE_MODEL=${BASE_MODEL:-"$PWD/models/Qwen2.5-1.5B-Instruct"}
DATA_DIR=${DATA_DIR:-"$PWD/data/game24"}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"game24-qwen2.5-1.5b-grpo"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=TinyZero \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null 2>&1 | tee game24_train.log
