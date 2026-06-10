#!/bin/bash
# Generate SFT training data for Game-of-24.
# Reads data/game24/train.parquet and writes data/sft_game24/{train,val}.parquet.
# Run prepare_data_game24.sh first.
#
# Usage (from project root):
#   bash scripts/generate_sft_data_game24.sh

set -e

INPUT=${INPUT:-"$PWD/data/game24/train.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PWD/data/sft_game24"}

# This machine uses a uv venv (py3.11), not the conda `zero` env from the original README.
PYTHON=${PYTHON:-"/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311/bin/python"}

"$PYTHON" examples/data_preprocess/generate_sft_game24.py \
    --input_parquet "$INPUT" \
    --output_dir    "$OUTPUT_DIR"

echo ""
echo "Done. SFT data saved to $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"
