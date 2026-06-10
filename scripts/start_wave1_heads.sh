#!/bin/bash
# Start 4 isolated Ray heads (2 GPUs each) for the Wave-1 GRPO sweep. Each head
# carries VERL_DISABLE_FLASH_ATTN_CE=1 and the arm's GAME24_REWARD so workers inherit
# them. Arms are launched separately (one tracked job each) connecting via RAY_ADDRESS.
#
#   arm0 CONTROL    GPUs 0,1  port 6379  sparse
#   arm1 N8         GPUs 2,3  port 6400  sparse
#   arm2 SHAPED     GPUs 4,5  port 6420  shaped
#   arm3 STAGED_P1  GPUs 6,7  port 6440  shaped
set -uo pipefail
RAY=/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311/bin/ray
$RAY stop --force >/dev/null 2>&1 || true; sleep 3
rm -rf /tmp/ray_w1_* 2>/dev/null || true
start_head () {  # gpus port reward
  CUDA_VISIBLE_DEVICES=$1 VERL_DISABLE_FLASH_ATTN_CE=1 GAME24_REWARD=$3 \
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/workspace/akide/models/huggingface \
    $RAY start --head --num-gpus=2 --port=$2 --dashboard-port=$(($2+1)) \
    --temp-dir=/tmp/ray_w1_$2 >/tmp/head_$2.log 2>&1
  echo "head port $2 (gpus $1, reward $3): rc=$? $(grep -o "ray start --address='[^']*'" /tmp/head_$2.log | head -1)"
}
start_head 0,1 6379 sparse
start_head 2,3 6400 sparse
start_head 4,5 6420 shaped
start_head 6,7 6440 shaped
echo "IP=$(hostname -I | awk '{print $1}')"
