#!/bin/bash
# Download and preprocess Countdown dataset (3-4 numbers → any target number)
# Source: Jiayi-Pan/Countdown-Tasks-3to4 on HuggingFace
#
# Usage (from project root):
#   bash scripts/prepare_data_countdown.sh
#
# Optional overrides:
#   TRAIN_SIZE=5000 TEST_SIZE=512 bash scripts/prepare_data_countdown.sh

set -e

DATA_DIR=${DATA_DIR:-"$PWD/data/countdown"}
TRAIN_SIZE=${TRAIN_SIZE:-10000}
TEST_SIZE=${TEST_SIZE:-1024}

# This machine uses a uv venv (py3.11), not the conda `zero` env from the original README.
PYTHON=${PYTHON:-"/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311/bin/python"}
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
export HF_HOME=${HF_HOME:-"/mnt/workspace/akide/models/huggingface"}

echo "Saving dataset to: $DATA_DIR"
mkdir -p "$DATA_DIR"

"$PYTHON" examples/data_preprocess/countdown.py \
    --local_dir "$DATA_DIR" \
    --train_size "$TRAIN_SIZE" \
    --test_size "$TEST_SIZE" \
    --template_type qwen-instruct

echo ""
echo "Done. Files:"
ls -lh "$DATA_DIR"
