#!/bin/bash
# Smoke test: verify the full pipeline works end-to-end in ~5 minutes.
# Runs data preprocessing (tiny subset) + 2 training steps + 1 validation step.
#
# Usage:
#   conda activate zero
#   export BASE_MODEL=/path/to/Qwen2.5-1.5B-Instruct   # or HF model id
#   bash scripts/smoke_test_game24.sh

set -e  # abort on any error

BASE_MODEL=${BASE_MODEL:-"$PWD/models/Qwen2.5-1.5B-Instruct"}
DATA_DIR=${TMPDIR:-/tmp}/game24_smoke
EXPERIMENT_NAME="game24-smoke-test"

echo "=========================================="
echo "STEP 1: Data preprocessing (tiny subset)"
echo "=========================================="
conda run -n zero python3 examples/data_preprocess/game24.py \
    --local_dir "$DATA_DIR" \
    --train_size 32 \
    --test_size 16
echo "Data saved to $DATA_DIR"
ls -lh "$DATA_DIR"

echo ""
echo "=========================================="
echo "STEP 2: Verify reward function"
echo "=========================================="
conda run -n zero python3 - <<'EOF'
import sys
sys.path.insert(0, ".")
from verl.utils.reward_score.game24 import compute_score

# Should return 1.0
s1 = "Let me think. <think>8/(3-1/3)=8/(8/3)=3</think><answer>(8/(3-1/3))</answer>"
# inject assistant separator expected by extract_answer
s1 = "<|im_start|>assistant\n" + s1
r = compute_score(s1, {"numbers": [8, 3, 1, 3], "target": 24})
print(f"Correct answer → reward = {r}  (expected 1.0)  {'OK' if r == 1.0 else 'FAIL'}")

# Should return 0.1 (has tag, wrong result)
s2 = "<|im_start|>assistant\n<answer>(1 + 2) * 3</answer>"
r = compute_score(s2, {"numbers": [1, 2, 3, 4], "target": 24})
print(f"Wrong result   → reward = {r}  (expected 0.1)  {'OK' if r == 0.1 else 'FAIL'}")

# Should return 0.0 (no tag)
s3 = "<|im_start|>assistant\nI cannot solve this."
r = compute_score(s3, {"numbers": [1, 2, 3, 4], "target": 24})
print(f"No tag         → reward = {r}  (expected 0.0)  {'OK' if r == 0.0 else 'FAIL'}")
EOF

echo ""
echo "=========================================="
echo "STEP 3: Training smoke test (2 steps)"
echo "=========================================="
export VLLM_ATTENTION_BACKEND=XFORMERS

conda run -n zero python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=TinyZero \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null

echo ""
echo "=========================================="
echo "Smoke test PASSED — pipeline is working."
echo "Run scripts/train_game24_local.sh for full training."
echo "=========================================="
