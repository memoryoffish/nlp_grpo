#!/bin/bash
# Start ONE isolated Ray head for a sweep arm.  Usage: start_head.sh <gpus> <port> <reward>
#   <gpus>   e.g. 0,1   (or a single GPU: 0)
#   <port>   e.g. 6379
#   <reward> sparse | shaped
# Workers inherit VERL_DISABLE_FLASH_ATTN_CE and GAME24_REWARD from this head's env.
set -uo pipefail
RAY=/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311/bin/ray
gpus=$1; port=$2; reward=${3:-sparse}
ngpu=$(awk -F, '{print NF}' <<< "$gpus")
CUDA_VISIBLE_DEVICES=$gpus VERL_DISABLE_FLASH_ATTN_CE=1 GAME24_REWARD=$reward \
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/workspace/akide/models/huggingface \
  $RAY start --head --num-gpus=$ngpu --port=$port --dashboard-port=$((port+1)) \
  --temp-dir=/tmp/ray_h_$port >/tmp/head_$port.log 2>&1
echo "head $port gpus=$gpus reward=$reward rc=$? $(grep -o "ray start --address='[^']*'" /tmp/head_$port.log | head -1)"
