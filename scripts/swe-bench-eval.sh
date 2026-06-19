#!/usr/bin/env bash
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
#   WORKERS      — Parallel workers.         Default: 3
#   DATASET_NAME — SWE-bench dataset.        Default: princeton-nlp/SWE-bench_Verified

set -euo pipefail

PREDS_PATH="${PREDS_PATH:-trajectories/preds.json}"
OUTPUT_DIR="${OUTPUT_DIR:-run-eval}"
RUN_ID="${RUN_ID:-eval}"
WORKERS="${WORKERS:-3}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"

echo "[swe-bench-eval] Starting evaluation"
echo "  preds_path   : ${PREDS_PATH}"
echo "  output_dir   : ${OUTPUT_DIR}"
echo "  run_id       : ${RUN_ID}"
echo "  workers      : ${WORKERS}"
echo "  dataset_name : ${DATASET_NAME}"

mkdir -p "${OUTPUT_DIR}"

uv run python -m swebench.harness.run_evaluation \
    --dataset_name   "${DATASET_NAME}" \
    --predictions_path "${PREDS_PATH}" \
    --max_workers    "${WORKERS}" \
    --run_id         "${RUN_ID}" \
    --output_dir     "${OUTPUT_DIR}"

echo "[swe-bench-eval] Done.  Output: ${OUTPUT_DIR}"
