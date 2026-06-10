#!/bin/bash
# Evaluate a trained checkpoint on the game24 test set (in-distribution).
# Also supports the hallucination set to check if the model fabricates solutions.
#
# Usage (from project root):
#   STEP=200 bash scripts/eval_game24.sh
#
# Include hallucination set:
#   STEP=200 HALLUC=1 bash scripts/eval_game24.sh
#
# Override checkpoint path directly:
#   CKPT=checkpoints/TinyZero/my-run/actor/global_step_300 bash scripts/eval_game24.sh

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

DATA_DIR=${DATA_DIR:-"$PWD/data/game24"}
OUTPUT=${OUTPUT:-"$PWD/results/eval_game24_step${STEP}.json"}

DATA_ARGS="$DATA_DIR/test.parquet"
if [ -n "${HALLUC}" ]; then
    DATA_ARGS="$DATA_ARGS $DATA_DIR/hallucination.parquet"
fi

PYTHON=${PYTHON:-"/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311/bin/python"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
BATCH_SIZE=${BATCH_SIZE:-8}

echo "Model : $MODEL"
echo "Data  : $DATA_ARGS"
echo "Output: $OUTPUT"
echo ""

"$PYTHON" examples/eval_ood.py \
    --model  "$MODEL" \
    --data   $DATA_ARGS \
    --output "$OUTPUT" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"
