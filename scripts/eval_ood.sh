#!/bin/bash
# OOD evaluation: evaluate a trained game24 checkpoint on countdown test set.
#
# Usage (from project root):
#   STEP=200 bash scripts/eval_ood.sh
#
# Override checkpoint path directly:
#   CKPT=checkpoints/TinyZero/my-run/actor/global_step_300 bash scripts/eval_ood.sh
#
# Evaluate on both game24 and countdown simultaneously:
#   STEP=200 EXTRA_DATA=$PWD/data/game24/test.parquet bash scripts/eval_ood.sh

set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"game24-qwen2.5-1.5b-grpo-local"}
STEP=${STEP:-""}

if [ -n "${CKPT}" ]; then
    MODEL="${CKPT}"
elif [ -n "${STEP}" ]; then
    MODEL="$PWD/checkpoints/TinyZero/${EXPERIMENT_NAME}/actor/global_step_${STEP}"
else
    echo "ERROR: set STEP=<n> or CKPT=<path>"
    exit 1
fi

OOD_DATA=${OOD_DATA:-"$PWD/data/countdown/test.parquet"}
OUTPUT=${OUTPUT:-"$PWD/results/ood_eval_step${STEP}.json"}

# build data list: always include countdown, optionally add extra files
DATA_ARGS="$OOD_DATA"
if [ -n "${EXTRA_DATA}" ]; then
    DATA_ARGS="$DATA_ARGS $EXTRA_DATA"
fi

PYTHON=${PYTHON:-"/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311/bin/python"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
BATCH_SIZE=${BATCH_SIZE:-8}

echo "Model   : $MODEL"
echo "OOD data: $DATA_ARGS"
echo "Output  : $OUTPUT"
echo ""

"$PYTHON" examples/eval_ood.py \
    --model  "$MODEL" \
    --data   $DATA_ARGS \
    --output "$OUTPUT" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"
