#!/usr/bin/env bash
# scripts/mini-swe-bench-batch.sh
#
# Run mini-swe-agent on a batch of SWE-bench tasks.
# All parameters are configurable via environment variables so this script
# can be called from the Airflow DAG or directly from the command line.
#
# Usage (direct):
#   SUBSET=verified SPLIT=test TASK_SLICE=0:5 WORKERS=3 OUTPUT_DIR=./out \
#       bash scripts/mini-swe-bench-batch.sh
#
# Environment variables:
#   SUBSET      — Dataset subset.  Default: verified
#   SPLIT       — Dataset split.   Default: test
#   MODEL       — Model ID.        Default: nebius/moonshotai/Kimi-K2.6
#   TASK_SLICE  — Python slice.    Default: 0:3
#   WORKERS     — Parallel workers.Default: 3
#   COST_LIMIT  — Per-task USD cap.Default: 0 (no limit)
#   OUTPUT_DIR  — Output directory.Default: trajectories/

set -euo pipefail

SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
MODEL="${MODEL:-nebius/moonshotai/Kimi-K2.6}"
TASK_SLICE="${TASK_SLICE:-0:3}"
WORKERS="${WORKERS:-3}"
COST_LIMIT="${COST_LIMIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-trajectories}"

echo "[mini-swe-bench-batch] Starting agent run"
echo "  subset     : ${SUBSET}"
echo "  split      : ${SPLIT}"
echo "  model      : ${MODEL}"
echo "  task_slice : ${TASK_SLICE}"
echo "  workers    : ${WORKERS}"
echo "  cost_limit : ${COST_LIMIT}"
echo "  output_dir : ${OUTPUT_DIR}"

CMD=(
    uv run mini-extra swebench
    --subset  "${SUBSET}"
    --split   "${SPLIT}"
    --model   "${MODEL}"
    --slice   "${TASK_SLICE}"
    --workers "${WORKERS}"
    --yolo
    -o        "${OUTPUT_DIR}"
)

if [ "${COST_LIMIT}" != "0" ] && [ "${COST_LIMIT}" != "0.0" ]; then
    CMD+=(--cost-limit "${COST_LIMIT}")
fi

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

echo "[mini-swe-bench-batch] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[mini-swe-bench-batch] Done.  Output: ${OUTPUT_DIR}"
