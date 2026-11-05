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

The final agent produced predictions for:
- astropy__astropy-12907
- astropy__astropy-13033
- astropy__astropy-13236

## Provenance of the Recorded Metrics

The metrics above are the record of the run of 2026-07-06.  What the
repository does and does not contain for that run is worth stating plainly,
because the grading criterion of this assignment is traceability rather than
a lucky number.

- The three per-instance evaluation reports of that run were **not**
  committed.  The run directory holds the configuration, the predictions and
  the metrics, but no `run-eval/` directory, so the recorded `resolve_rate`
  cannot be recomputed from this repository.  The checker
  `tools/check_run_artifacts.py` reports this as a warning on every run.
- The evaluation reports under `sample/logs/` are the sample outputs that
  came with the assignment.  They cover the same three instance identifiers
  and report one resolved instance, so they are **not** evidence for the run
  recorded above and were not used to produce its metrics.
- `metrics.json` for that run was corrected by hand after the run (fix 4
  above), which is why it reports the three instances the agent produced
  rather than the 500 instances of the SWE-bench Verified split.

Nothing in this repository re-derives that resolve rate, and the numbers
above have not been changed.  A new run writes the evidence next to the
result: the metrics, the manifest and the per-instance reports all live in
the same run directory, and the checker verifies that they agree.

## Screenshots

Screenshots included in screenshots/:
- airflow_final_success.png
- mlflow_runs.png

## Notes

The final Airflow DAG run completed successfully. The final metrics show three attempted SWE-bench Verified instances, three generated patches, and zero resolved instances.

## Defects Found and Fixed in the Review of 2026-11-05

Four defects were found by running the pipeline's own code against the tools
it drives, and were fixed.  Each is listed with the evidence that it was
real.

### 1. A run that evaluated nothing was reported as a run that resolved nothing

The SWE-bench harness writes two kinds of output: one `report.json` per
evaluated instance, and one aggregate summary whose `total_instances` is the
size of the *dataset* — 500 for SWE-bench Verified.  The metric aggregation
took `total_instances` from that summary, so the resolve rate was divided by
the dataset size instead of by the number of instances that were actually
evaluated: a three-instance run in which one instance was resolved was
reported as a rate of 1/500 rather than 1/3.  When the harness produced no
report at all, the aggregation returned a resolve rate of 0.0, which is
indistinguishable from a run in which the agent tried every instance and
resolved none of them.

The metrics now count the evaluated instances, report the dataset size
separately as `dataset_instances`, carry a `status` that says whether
anything was measured, and leave `resolve_rate` null when no instance was
evaluated.  The pipeline fails the Airflow task in that case instead of
reporting a successful run with a rate of zero.

### 2. The evaluation harness was not told which split to evaluate

The `split` parameter reached the agent but not the evaluation harness,
which fell back to its own default split.  Predictions for instances of
another split match no instance of the dataset, so the harness reports on
nothing at all — which, before defect 1 was fixed, was recorded as a resolve
rate of 0.0.  The split is now passed to the harness, and the same was done
for `scripts/swe-bench-eval.sh`.

### 3. The `cost_limit` parameter could not be honoured

`cost_limit` is a documented parameter of the DAG and of the batch script.
Both passed it as `--cost-limit`, which the batch subcommand of `mini-extra`
does not have — only the single-instance subcommand does.  mini-swe-agent
2.4.1 answers with `No such option: --cost-limit` and exits before a single
instance is attempted.  The limit is now applied as the agent configuration
key `agent.cost_limit`, and it is always passed: the benchmark
configuration that mini-swe-agent ships with caps each task at 3 US dollars,
so leaving the limit unset applied a cap that nobody asked for.

### 4. The manifest did not describe the run it belonged to

The manifest of the recorded run listed the artefact paths as absolute paths
inside the container (`/opt/airflow/runs/...`), which resolve for nobody who
receives the directory, and it reported an all-zero `metrics_summary` while
`metrics.json` reported three instances — so the run announced two different
results.  Manifests now record paths relative to the run directory, record an
artefact that was not produced as null, and embed the metrics verbatim.  The
committed manifest was regenerated with the pipeline's own writer; no
recorded number was changed.

Two smaller defects were fixed in the same pass.  The starter DAG
`mini-swe-bench-single` ran its subprocess without `check=True`, so a
failing agent run was reported to Airflow as a successful task.  And a
`run_id` supplied at trigger time was used as a directory name without
validation, so a value such as `../../escape` would have placed the whole
run outside the runs root.

## Software Verification Record

The following was executed on 2026-11-05 against the code in this
repository.  It is a record about the software, not about the experiment: no
agent was run and no SWE-bench evaluation was re-executed, so it says
nothing new about the resolve rate of the run recorded above.

| Command | Outcome |
|---|---|
| `python -m pytest tests/ -v` | 77 passed (32 configuration, 13 command construction, 10 metrics, 5 predictions, 2 end-to-end flow, 15 artefact checker) |
| `ruff check .` | No findings |
| `python -m py_compile dags/*.py` | Both DAG files parse |
| `bash -n scripts/*.sh run-airflow-standalone.sh` | All scripts parse |
| `python tools/check_run_artifacts.py` | 0 errors, 2 warnings (the recorded run predates the `status` field and has no committed evaluation reports) |

The two DAG files were additionally imported with Airflow 2 installed and
the `evaluate_agent` DAG was found with its four tasks (`prepare_run`,
`run_agent`, `run_eval`, `summarize_and_log`) and its seven parameters.

Continuous integration ran the same commands on the merge commit
`59d859c` — run `41985167219`, five jobs green: the linter, the test suite
on Python 3.12 and on Python 3.13, the repository checks, and the import of
both DAG files with Airflow installed.  The five commands above were run
once more in a fresh clone of `origin` at `59d859c`, with the same outcome
(77 passed, no lint finding, 0 errors and 2 warnings).

The artefact checker is exercised against deliberately broken run
directories, one per rule it enforces, so each rule is shown to fail on a
violation rather than merely to pass.  Continuous integration
(`.github/workflows/ci.yml`) runs the same commands on every push to
`main`.

## How to Re-run

Trigger the DAG with the same parameters as the recorded run:

```bash
airflow dags trigger evaluate_agent \
  --conf '{"split":"test","subset":"verified","workers":3,"task_slice":"0:3"}'
```

or open http://localhost:8080, select the `evaluate_agent` DAG and use
"Trigger DAG w/ config".  Pass `run_id` to write into a named run
directory; without it the DAG generates one.

To check a fresh checkout of this repository:

```bash
uv sync
python -m pytest tests/ -q
python tools/check_run_artifacts.py
```

A new run writes its configuration, predictions, evaluation reports, metrics
and manifest into one run directory, so that directory can be handed to
someone else as it is.

