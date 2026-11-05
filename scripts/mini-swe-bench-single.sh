#!/usr/bin/env bash
#
# scripts/mini-swe-bench-single.sh
#
# Run mini-swe-agent on a single SWE-bench instance.
#
# Usage (direct):
#   bash scripts/mini-swe-bench-single.sh
#
# Environment variables:
#   SUBSET      — Dataset subset.  Default: verified
#   SPLIT       — Dataset split.   Default: test
#   MODEL       — Model ID.        Default: nebius/moonshotai/Kimi-K2.6
#   COST_LIMIT  — Per-task USD cap.Default: 0 (no limit)
#   INSTANCE    — Instance to run.  Default: sympy__sympy-15599
#   OUTPUT      — Trajectory file.  Default: trajectory.json

set -euo pipefail

SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
MODEL="${MODEL:-nebius/moonshotai/Kimi-K2.6}"
COST_LIMIT="${COST_LIMIT:-0}"
INSTANCE="${INSTANCE:-sympy__sympy-15599}"
OUTPUT="${OUTPUT:-trajectory.json}"

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

# The single-instance subcommand does accept --cost-limit, unlike the batch
# subcommand in mini-swe-bench-batch.sh.
mini-extra swebench-single \
    --subset "${SUBSET}" \
    --split "${SPLIT}" \
    --model "${MODEL}" \
    --cost-limit "${COST_LIMIT}" \
    -i "${INSTANCE}" \
    -o "${OUTPUT}"
