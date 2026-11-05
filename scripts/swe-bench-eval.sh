#!/usr/bin/env bash
#
# scripts/swe-bench-eval.sh
#
# Run the SWE-bench evaluation harness on a predictions file.
# All parameters are configurable via environment variables.
#
# Usage (direct):
#   PREDS_PATH=runs/my-run/run-agent/preds.json \
#   OUTPUT_DIR=runs/my-run/run-eval \
#   RUN_ID=my-run \
#       bash scripts/swe-bench-eval.sh
#
# Environment variables:
#   PREDS_PATH   — Path to preds.json.       Required.
#   OUTPUT_DIR   — Where to write eval logs. Default: run-eval/
#   RUN_ID       — Identifier for this eval. Default: eval
#   SPLIT        — Dataset split to evaluate. Default: test
#   WORKERS      — Parallel workers.         Default: 3
#   DATASET_NAME — SWE-bench dataset.        Default: princeton-nlp/SWE-bench_Verified

set -euo pipefail

PREDS_PATH="${PREDS_PATH:-trajectories/preds.json}"
OUTPUT_DIR="${OUTPUT_DIR:-run-eval}"
RUN_ID="${RUN_ID:-eval}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-3}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"

echo "[swe-bench-eval] Starting evaluation"
echo "  preds_path   : ${PREDS_PATH}"
echo "  output_dir   : ${OUTPUT_DIR}"
echo "  run_id       : ${RUN_ID}"
echo "  split        : ${SPLIT}"
echo "  workers      : ${WORKERS}"
echo "  dataset_name : ${DATASET_NAME}"

mkdir -p "${OUTPUT_DIR}"

# The split is passed explicitly.  The harness otherwise falls back to its
# own default split, and a prediction for an instance of a different split
# matches nothing, so the harness would report on no instance at all.
uv run python -m swebench.harness.run_evaluation \
    --dataset_name   "${DATASET_NAME}" \
    --predictions_path "${PREDS_PATH}" \
    --split          "${SPLIT}" \
    --max_workers    "${WORKERS}" \
    --run_id         "${RUN_ID}" \
    --report_dir     "${OUTPUT_DIR}"

echo "[swe-bench-eval] Done.  Output: ${OUTPUT_DIR}"
