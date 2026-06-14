#!/usr/bin/env bash
# Parallel evaluation of all ablation models on the COMPLIANT test (ttc idx900-999).
# 4 eval_compare instances, one per GPU (1-4); eval is plain torch inference (no Ray) so
# concurrent instances don't conflict. Each writes its own JSON; merge afterward.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311/bin/python
CK="$ROOT/checkpoints/TinyZero"
D="$ROOT/data/game24_official"
export HF_ENDPOINT=https://hf-mirror.com
COMMON="--test $D/test.parquet --train $D/train.parquet --halluc $D/hallucination.parquet \
  --countdown $ROOT/data/countdown/test.parquet --countdown_n 256 --bon 16 --batch_size 16 --max_new_tokens 320"

run () {  # $1=gpu $2=out  $3..=models
  local gpu=$1 out=$2; shift 2
  local margs=""; for m in "$@"; do margs="$margs --model $m"; done
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_compare.py $margs $COMMON \
    --out_json "$ROOT/results/$out.json" --out_md "$ROOT/results/$out.md" > "/tmp/$out.log" 2>&1
}

run 1 cmp_g1 \
  "base:/mnt/workspace/akide/models/Qwen2.5-1.5B-Instruct" \
  "SFT-multi:$CK/game24-sft-official/global_step_296" \
  "SFT-single:$CK/game24-sft-single/global_step_148" &
run 2 cmp_g2 \
  "C-multi-n4-lr1e6:$CK/abl-C-multi-n4-sparse-lr1e6/actor/global_step_40" \
  "A1-n8:$CK/abl-A1-n8/actor/global_step_40" \
  "A2-single:$CK/abl-A2-single/actor/global_step_40" &
run 3 cmp_g3 \
  "A3-shaped:$CK/abl-A3-shaped/actor/global_step_40" \
  "A4a-lr5e7:$CK/abl-A4a-lr5e7/actor/global_step_40" \
  "A4b-lr2e6:$CK/abl-A4b-lr2e6/actor/global_step_40" &
run 4 cmp_g4 \
  "A4c-lr5e6:$CK/abl-A4c-lr5e6/actor/global_step_40" \
  "R1-puregrpo:$CK/abl-R1-puregrpo/actor/global_step_40" &
wait
echo "===== ALL COMPLIANT EVAL DONE ($(date '+%T')) ====="
