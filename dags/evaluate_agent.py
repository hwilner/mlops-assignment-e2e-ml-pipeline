"""Airflow DAG: evaluate_agent — End-to-end SWE-bench evaluation pipeline.

This DAG orchestrates the full evaluation workflow for mini-swe-agent on
SWE-bench tasks.  It implements the four-phase pipeline described in the
assignment README:

    prepare_run  →  run_agent  →  run_eval  →  summarize_and_log

Each run is fully reproducible from its ``runs/<run-id>/`` directory, which
contains the configuration, agent trajectories, evaluation logs, metrics, and
a manifest pointing to all artefacts.

The DAG itself is deliberately thin.  Everything that can be decided without
a scheduler lives in the ``pipeline`` package next to this file, where it is
unit-tested: the run configuration and its validation
(:mod:`pipeline.config`), the two command lines
(:mod:`pipeline.evaluation`), and the metrics
(:mod:`pipeline.metrics`).  This module only wires the tasks together and
talks to Airflow and MLflow.

Airflow parameters (set via the "Trigger DAG w/ config" UI or the CLI):

    Required:
        split    (str)  — SWE-bench dataset split, e.g. "test" or "dev".
        subset   (str)  — Dataset subset, e.g. "verified" or "lite".
        workers  (int)  — Parallel workers for agent and evaluation steps.

    Optional (with sensible defaults):
        model       (str)  — Model ID passed to mini-swe-agent.
                             Default: "nebius/moonshotai/Kimi-K2.6".
        task_slice  (str)  — Python slice string to select a subset of tasks,
                             e.g. "0:5".  Default: "0:3".
        run_id      (str)  — Explicit run identifier.  If omitted, a timestamp-
                             based ID is generated automatically.
        cost_limit  (float)— Per-task cost limit in USD.  0 means no limit.
                             Default: 0.

Example CLI trigger::

    airflow dags trigger evaluate_agent \\
        --conf '{"split":"test","subset":"verified","workers":3,"task_slice":"0:5"}'

Example Airflow UI trigger:
    Open http://localhost:8080, find "evaluate_agent", click the ▶ button,
    paste the JSON config above, and click "Trigger".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.models.param import Param

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# The DAG file is mounted at /opt/airflow/dags/ in Docker, but the project
# lives at /mlops-assignment, so the project root is taken from the
# environment and only falls back to the location of this file.
#
# docker-compose.yaml sets:
#   PROJECT_ROOT=/mlops-assignment
#   RUNS_ROOT=/opt/airflow/runs
#
# Local / standalone mode: both env vars are unset, so the fallback resolves
# to the repo root via __file__ as before.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", str(PROJECT_ROOT / "runs")))

# The pipeline package is imported rather than duplicated here.  Airflow
# imports this file with its own working directory, so the project root has
# to be on the import path explicitly; the import therefore comes after the
# path setup above.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (  # noqa: E402  (import follows the path setup)
    DEFAULT_COST_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_TASK_SLICE,
    DEFAULT_WORKERS,
    build_run_config,
    prepare_run_dir,
)
from pipeline.evaluation import (  # noqa: E402  (import follows the path setup)
    build_agent_command,
    build_eval_command,
)
from pipeline.metrics import (  # noqa: E402  (import follows the path setup)
    METRICS_STATUS_OK,
    collect_metrics,
    write_manifest,
)
from pipeline.predictions import (  # noqa: E402  (import follows the path setup)
    read_predictions,
)

# ===========================================================================
# Helper functions
# ===========================================================================

def run_agent_batch(run_config: dict[str, Any], run_dir: Path) -> Path:
    """Execute mini-swe-agent on the configured SWE-bench subset.

    Calls ``mini-extra swebench`` via ``uv run`` with the parameters from
    *run_config*.  Trajectories and ``preds.json`` are written to
    ``run_dir/run-agent/``.

    Args:
        run_config: Normalised run configuration dictionary.
        run_dir:    Path to the run directory (``runs/<run-id>/``).

    Returns:
        Path to the ``run-agent/`` sub-directory containing ``preds.json``
        and per-instance trajectory files.

    Raises:
        subprocess.CalledProcessError: If the agent process exits with a
            non-zero return code.
    """
    agent_out_dir = run_dir / "run-agent"
    cmd = build_agent_command(run_config, agent_out_dir)

    env = {
        **os.environ,
        "MSWEA_COST_TRACKING": "ignore_errors",
    }

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    return agent_out_dir


def run_swebench_eval(
    run_config: dict[str, Any],
    preds_path: Path,
    run_dir: Path,
) -> Path:
    """Run the SWE-bench evaluation harness on the agent predictions.

    Calls ``python -m swebench.harness.run_evaluation`` with the predictions
    produced by ``run_agent_batch()``.  Evaluation logs and per-instance
    reports are written to ``run_dir/run-eval/``.

    Args:
        run_config:  Normalised run configuration dictionary.
        preds_path:  Path to the ``preds.json`` file produced by the agent.
        run_dir:     Path to the run directory (``runs/<run-id>/``).

    Returns:
        Path to the ``run-eval/`` sub-directory containing evaluation logs
        and per-instance ``report.json`` files.

    Raises:
        subprocess.CalledProcessError: If the evaluation process exits with a
            non-zero return code.
    """
    eval_out_dir = run_dir / "run-eval"
    cmd = build_eval_command(run_config, preds_path, eval_out_dir)

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return eval_out_dir


def log_mlflow_run(
    run_config: dict[str, Any],
    metrics: dict[str, Any],
    artifact_uri: str,
) -> None:
    """Log parameters, metrics, and artifact reference to MLflow.

    Creates a new MLflow run under the experiment named after the model,
    logs all run-configuration keys as parameters, all metrics, and records
    the local artifact path as a tag so the run can be reconstructed.

    Args:
        run_config:   Normalised run configuration dictionary.
        metrics:      Aggregated metrics as returned by ``collect_metrics()``.
        artifact_uri: Local path or remote URI for the run artefacts directory.

    Note:
        MLflow tracking URI is read from the ``MLFLOW_TRACKING_URI`` environment
        variable.  Defaults to ``http://localhost:5000`` if not set.
    """
    try:
        import mlflow  # type: ignore[import]
    except ImportError:
        print(
            "[summarize_and_log] mlflow not installed — skipping MLflow logging. "
            "Install with: uv add mlflow"
        )
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # Use the model name as the experiment name for easy comparison.
    experiment_name = f"swebench-{run_config['model'].replace('/', '__')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_config["run_id"]):
        # Log all configuration keys as MLflow parameters.
        for key, value in run_config.items():
            mlflow.log_param(key, value)

        # Log all numeric metrics.  A metric that could not be computed is
        # skipped rather than logged as zero.
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, float(value))

        # Tag the run with the artifact location for easy retrieval.
        mlflow.set_tag("artifact_uri", artifact_uri)
        mlflow.set_tag("run_id", run_config["run_id"])
        # The status says whether the metrics describe a measured result.
        mlflow.set_tag("metrics_status", str(metrics.get("status", "UNKNOWN")))

    print(
        f"[summarize_and_log] MLflow run logged to {tracking_uri} "
        f"under experiment '{experiment_name}'"
    )


# ===========================================================================
# DAG definition
# ===========================================================================

@dag(
    dag_id="evaluate_agent",
    description=(
        "End-to-end SWE-bench evaluation: run mini-swe-agent on a configurable "
        "subset, evaluate with SWE-bench harness, and log results to MLflow."
    ),
    start_date=datetime(2024, 1, 1, tzinfo=UTC),
    schedule=None,       # manual / API-triggered only
    catchup=False,
    tags=["swebench", "mlops", "evaluation"],
    params={
        # Required params — must be supplied at trigger time.
        "split": Param(
            default="test",
            type="string",
            description="SWE-bench dataset split (e.g. 'test' or 'dev').",
        ),
        "subset": Param(
            default="verified",
            type="string",
            description="Dataset subset (e.g. 'verified' or 'lite').",
        ),
        "workers": Param(
            default=DEFAULT_WORKERS,
            type="integer",
            description="Number of parallel workers for agent and eval steps.",
        ),
        # Optional params — all have sensible defaults.
        "model": Param(
            default=DEFAULT_MODEL,
            type="string",
            description="Model ID passed to mini-swe-agent.",
        ),
        "task_slice": Param(
            default=DEFAULT_TASK_SLICE,
            type="string",
            description=(
                "Python slice string selecting a subset of tasks, e.g. '0:5'. "
                "Use '0:3' for a quick smoke test."
            ),
        ),
        "run_id": Param(
            default="",
            type="string",
            description=(
                "Explicit run identifier.  Leave empty to auto-generate a "
                "timestamp-based ID."
            ),
        ),
        "cost_limit": Param(
            default=DEFAULT_COST_LIMIT,
            type="number",
            description="Per-task cost limit in USD.  0 = no limit.",
        ),
    },
)
def evaluate_agent_dag():
    """DAG: evaluate_agent — configurable SWE-bench evaluation pipeline.

    Trigger via the Airflow UI ("Trigger DAG w/ config") or the CLI::

        airflow dags trigger evaluate_agent \\
            --conf '{"split":"test","subset":"verified","workers":3}'

    The four tasks run in sequence:

    1. ``prepare_run``      — validate params, create run directory, write config.json.
    2. ``run_agent``        — execute mini-swe-agent, write trajectories + preds.json.
    3. ``run_eval``         — run SWE-bench harness, write logs + per-instance reports.
    4. ``summarize_and_log``— aggregate metrics, write metrics.json + manifest.json,
                              log everything to MLflow.
    """

    @task
    def prepare_run(**context) -> dict:
        """Validate Airflow params, create the run directory, and write config.json.

        Reads ``dag_run.conf`` (the JSON supplied at trigger time), applies
        defaults for optional keys, generates a ``run_id``, creates the
        ``runs/<run-id>/`` directory tree, and writes ``config.json``.

        Returns:
            The normalised run-configuration dictionary (JSON-serialisable),
            which is passed to downstream tasks via XCom.

        Raises:
            ValueError: If a parameter is missing or fails validation.
        """
        raw_params = context["dag_run"].conf or {}
        run_config = build_run_config(raw_params)
        run_dir = prepare_run_dir(run_config, RUNS_ROOT)
        print(
            f"[prepare_run] Run directory created: {run_dir}\n"
            f"[prepare_run] Config: {json.dumps(run_config, indent=2)}"
        )
        # Return a JSON-serialisable dict for XCom.
        return {**run_config, "_run_dir": str(run_dir)}

    @task
    def run_agent(run_config_xcom: dict) -> dict:
        """Execute mini-swe-agent on the configured SWE-bench subset.

        Reads the run configuration from XCom, calls ``run_agent_batch()``,
        and returns the path to the agent output directory.

        Args:
            run_config_xcom: Run configuration dict pushed by ``prepare_run``.

        Returns:
            Dictionary with keys ``agent_out_dir`` (str path) and the full
            ``run_config`` for downstream tasks.
        """
        run_dir = Path(run_config_xcom.pop("_run_dir"))
        run_config = run_config_xcom

        print(
            f"[run_agent] Starting mini-swe-agent\n"
            f"  model      : {run_config['model']}\n"
            f"  split      : {run_config['split']}\n"
            f"  subset     : {run_config['subset']}\n"
            f"  task_slice : {run_config['task_slice']}\n"
            f"  workers    : {run_config['workers']}\n"
            f"  cost_limit : {run_config['cost_limit']}\n"
            f"  output     : {run_dir / 'run-agent'}"
        )

        agent_out_dir = run_agent_batch(run_config, run_dir)

        print(f"[run_agent] Agent finished.  Output: {agent_out_dir}")
        return {
            **run_config,
            "_run_dir":       str(run_dir),
            "_agent_out_dir": str(agent_out_dir),
        }

    @task
    def run_eval(agent_xcom: dict) -> dict:
        """Run the SWE-bench evaluation harness on the agent predictions.

        Locates ``preds.json`` in the agent output directory, checks that it
        holds at least one prediction, and calls ``run_swebench_eval()``.

        Args:
            agent_xcom: Dictionary pushed by ``run_agent`` containing
                ``_run_dir`` and ``_agent_out_dir``.

        Returns:
            Dictionary with ``_eval_out_dir`` (str path) added for downstream
            tasks.

        Raises:
            FileNotFoundError: If ``preds.json`` does not exist in the agent
                output directory.
            ValueError: If ``preds.json`` is malformed or holds no
                prediction.
        """
        run_dir = Path(agent_xcom.pop("_run_dir"))
        agent_out_dir = Path(agent_xcom.pop("_agent_out_dir"))
        run_config = agent_xcom

        preds_path = agent_out_dir / "preds.json"
        predictions = read_predictions(preds_path)

        print(
            f"[run_eval] Starting SWE-bench evaluation\n"
            f"  predictions : {preds_path} ({len(predictions)} instances)\n"
            f"  output      : {run_dir / 'run-eval'}"
        )

        eval_out_dir = run_swebench_eval(run_config, preds_path, run_dir)

        print(f"[run_eval] Evaluation finished.  Output: {eval_out_dir}")
        return {
            **run_config,
            "_run_dir":      str(run_dir),
            "_eval_out_dir": str(eval_out_dir),
        }

    @task
    def summarize_and_log(eval_xcom: dict) -> None:
        """Aggregate metrics, write artefact files, and log to MLflow.

        Parses per-instance ``report.json`` files produced by the SWE-bench
        harness, aggregates them into ``metrics.json``, writes ``manifest.json``
        for full run reproducibility, and logs everything to MLflow.

        Args:
            eval_xcom: Dictionary pushed by ``run_eval`` containing
                ``_run_dir`` and ``_eval_out_dir``.

        Raises:
            AirflowException: If the evaluation produced no instance report.
                The run is written out first so the failure can be
                investigated, but the task is marked failed because nothing
                was measured.
        """
        run_dir = Path(eval_xcom.pop("_run_dir"))
        eval_out_dir = Path(eval_xcom.pop("_eval_out_dir"))
        run_config = eval_xcom

        # Aggregate metrics from per-instance report.json files.
        metrics = collect_metrics(eval_out_dir)

        # Write metrics.json.
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        # Write manifest.json.
        manifest_path = write_manifest(run_dir, run_config, metrics)

        print(
            f"[summarize_and_log] Metrics:\n{json.dumps(metrics, indent=2)}\n"
            f"[summarize_and_log] manifest written to: {manifest_path}"
        )

        # Log to MLflow.
        log_mlflow_run(run_config, metrics, artifact_uri=str(run_dir))

        if metrics.get("status") != METRICS_STATUS_OK:
            # The metrics were written and logged so the failure can be
            # investigated, but the task must fail: a run that measured
            # nothing is not a successful run.
            raise AirflowException(
                f"The evaluation measured no result (status "
                f"{metrics.get('status')!r}). The run directory is "
                f"{run_dir}. Notes: {'; '.join(metrics.get('notes', [])) or 'none'}."
            )

        print(
            f"[summarize_and_log] Run complete.\n"
            f"  run_id      : {run_config['run_id']}\n"
            f"  resolve_rate: {metrics.get('resolve_rate', 0):.1%}\n"
            f"  run_dir     : {run_dir}"
        )

    # Wire the tasks together in sequence.
    prepare_result = prepare_run()
    agent_result   = run_agent(prepare_result)
    eval_result    = run_eval(agent_result)
    summarize_and_log(eval_result)


# Instantiate the DAG so Airflow can discover it.
evaluate_agent_dag()
