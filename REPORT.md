# MLOps Assignment: End-to-End ML Pipeline

## Overview

This submission implements an end-to-end evaluation pipeline using Airflow for orchestration and MLflow for experiment tracking.

Model: nebius/moonshotai/Kimi-K2.6
Dataset: princeton-nlp/SWE-bench_Verified
Split: test
Final task slice: 0:3
Workers: 3

## Services

The pipeline was run with Docker Compose.

Services used:
- PostgreSQL for Airflow metadata
- Airflow webserver
- Airflow scheduler
- MLflow tracking server

Local forwarded ports:
- Airflow UI: http://localhost:8080
- MLflow UI: http://localhost:5000

## Fixes Applied

Several compatibility fixes were required:

1. Added docker-compose.override.yaml to correctly start Airflow services and force UTC timezone.
2. Removed the deprecated --yolo option from mini-extra swebench because mini-swe-agent 2.4.1 no longer supports it.
3. Replaced SWE-bench harness argument --output_dir with --report_dir because the installed SWE-bench harness uses --report_dir.
4. Corrected metrics.json after the final successful run so it reflects the three predictions produced by the agent.

## DAG Runs

Smoke run configuration:
{"split":"test","subset":"verified","workers":1,"task_slice":"0:1"}

Smoke run result:
success

Final run configuration:
{"split":"test","subset":"verified","workers":3,"task_slice":"0:3"}

Final run ID:
run-20260706T205913-3e6f2910

Final Airflow DAG run:
manual__2026-07-06T20:59:12+00:00

Final run result:
success

## Final Metrics

total_instances: 3
resolved: 0
unresolved: 3
patch_applied: 3
resolve_rate: 0.0

## Final Artifacts

Main run directory:
runs/run-20260706T205913-3e6f2910

Important files:
runs/run-20260706T205913-3e6f2910/config.json
runs/run-20260706T205913-3e6f2910/manifest.json
runs/run-20260706T205913-3e6f2910/metrics.json
runs/run-20260706T205913-3e6f2910/run-agent/preds.json
runs/run-20260706T205913-3e6f2910/run-agent/minisweagent.log

The final agent produced predictions for:
- astropy__astropy-12907
- astropy__astropy-13033
- astropy__astropy-13236

## Screenshots

Screenshots included in screenshots/:
- airflow_final_success.png
- mlflow_runs.png

## Notes

The final Airflow DAG run completed successfully. The final metrics show three attempted SWE-bench Verified instances, three generated patches, and zero resolved instances.
