#!/bin/bash
# Preprocess Game-of-24 dataset and save to ./data/game24
#
# Usage (from project root):
#   bash scripts/prepare_data_game24.sh
#
# Optional overrides:
#   DATA_DIR=./data/game24   bash scripts/prepare_data_game24.sh
#   TRAIN_SIZE=500           bash scripts/prepare_data_game24.sh

set -e

DATA_DIR=${DATA_DIR:-"$PWD/data/game24"}
TRAIN_SIZE=${TRAIN_SIZE:-1262}
TEST_SIZE=${TEST_SIZE:-100}
HALLUC_SIZE=${HALLUC_SIZE:-100}

# This machine uses a uv venv (py3.11), not the conda `zero` env from the original README.
PYTHON=${PYTHON:-"/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311/bin/python"}
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
export HF_HOME=${HF_HOME:-"/mnt/workspace/akide/models/huggingface"}

echo "Saving dataset to: $DATA_DIR"

"$PYTHON" examples/data_preprocess/game24.py \
    --local_dir "$DATA_DIR" \
    --train_size "$TRAIN_SIZE" \
    --test_size "$TEST_SIZE" \
    --halluc_size "$HALLUC_SIZE"

echo ""
echo "Done. Files:"
ls -lh "$DATA_DIR"
