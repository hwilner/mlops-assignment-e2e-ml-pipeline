"""Airflow DAG: evaluate_agent — End-to-end SWE-bench evaluation pipeline.

This DAG orchestrates the full evaluation workflow for mini-swe-agent on
SWE-bench tasks.  It implements the four-phase pipeline described in the
assignment README:

    prepare_run  →  run_agent  →  run_eval  →  summarize_and_log

Each run is fully reproducible from its ``runs/<run-id>/`` directory, which
contains the configuration, agent trajectories, evaluation logs, metrics, and
a manifest pointing to all artefacts.

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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.models.param import Param

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"
DEFAULT_TASK_SLICE = "0:3"
DEFAULT_WORKERS = 3
DEFAULT_COST_LIMIT = 0


# ===========================================================================
# Helper functions
# ===========================================================================

def build_run_config(params: dict[str, Any]) -> dict[str, Any]:
    """Build a normalised run configuration dictionary from raw Airflow params.

    Applies defaults for optional parameters and generates a ``run_id`` if one
    was not supplied.  The returned dict is the single source of truth for the
    entire DAG run and is written to ``config.json`` by ``prepare_run_dir``.

    Args:
        params: Raw Airflow ``dag_run.conf`` dictionary.  Must contain at
            minimum ``split``, ``subset``, and ``workers``.

    Returns:
        A fully populated run-configuration dictionary with keys:
            run_id, split, subset, workers, model, task_slice, cost_limit,
            created_at.

    Raises:
        ValueError: If any of the required keys (split, subset, workers) are
            missing from *params*.
    """
    for required_key in ("split", "subset", "workers"):
        if required_key not in params:
            raise ValueError(
                f"Required Airflow param '{required_key}' is missing. "
                "Trigger the DAG with at least: "
                '{"split": "test", "subset": "verified", "workers": 3}'
            )

    # Generate a deterministic run_id if not supplied.
    run_id = params.get("run_id") or (
        f"run-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-"
        f"{str(uuid.uuid4())[:8]}"
    )

    return {
        "run_id":      run_id,
        "split":       str(params["split"]),
        "subset":      str(params["subset"]),
        "workers":     int(params["workers"]),
        "model":       str(params.get("model", DEFAULT_MODEL)),
        "task_slice":  str(params.get("task_slice", DEFAULT_TASK_SLICE)),
        "cost_limit":  float(params.get("cost_limit", DEFAULT_COST_LIMIT)),
        "created_at":  datetime.utcnow().isoformat() + "Z",
    }


def prepare_run_dir(run_config: dict[str, Any]) -> Path:
    """Create the run directory tree and write ``config.json``.

    Creates the following layout under ``runs/<run-id>/``::

        runs/<run-id>/
            config.json          ← full run configuration
            run-agent/           ← populated by run_agent_batch()
            run-eval/            ← populated by run_swebench_eval()

    Args:
        run_config: Normalised run configuration as returned by
            ``build_run_config()``.

    Returns:
        The ``Path`` object pointing to ``runs/<run-id>/``.
    """
    run_dir = RUNS_ROOT / run_config["run_id"]

    # Create sub-directories.
    (run_dir / "run-agent").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=True)

    # Write the canonical config file.
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(run_config, indent=2))

    return run_dir


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

    cmd = [
        "uv", "run", "mini-extra", "swebench",
        "--subset",  run_config["subset"],
        "--split",   run_config["split"],
        "--model",   run_config["model"],
        "--slice",   run_config["task_slice"],
        "--workers", str(run_config["workers"]),
        "--yolo",
        "-o", str(agent_out_dir),
    ]

    # Append cost-limit only when a positive value is set.
    if run_config["cost_limit"] > 0:
        cmd += ["--cost-limit", str(run_config["cost_limit"])]

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

    cmd = [
        "uv", "run", "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name",   "princeton-nlp/SWE-bench_Verified",
        "--predictions_path", str(preds_path),
        "--max_workers",    str(run_config["workers"]),
        "--run_id",         run_config["run_id"],
        "--output_dir",     str(eval_out_dir),
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return eval_out_dir


def collect_metrics(eval_dir: Path) -> dict[str, Any]:
    """Parse evaluation output and aggregate per-run metrics.

    Walks the ``eval_dir`` tree looking for per-instance ``report.json``
    files (written by the SWE-bench harness) and aggregates them into a
    single metrics dictionary.  Also reads the top-level summary JSON if
    present.

    Args:
        eval_dir: Path to the ``run-eval/`` directory produced by
            ``run_swebench_eval()``.

    Returns:
        A dictionary with at minimum the following keys:
            total_instances   (int)  — number of evaluated instances.
            resolved          (int)  — number of resolved instances.
            resolve_rate      (float)— resolved / total_instances.
            unresolved        (int)  — number of unresolved instances.
            patch_applied     (int)  — instances where the patch applied.
    """
    # Collect all per-instance report.json files.
    report_files = list(eval_dir.rglob("report.json"))

    total = len(report_files)
    resolved = 0
    patch_applied = 0

    for rf in report_files:
        try:
            data = json.loads(rf.read_text())
            # Each report.json has one key (the instance id) mapping to a dict.
            for instance_data in data.values():
                if instance_data.get("resolved"):
                    resolved += 1
                if instance_data.get("patch_successfully_applied"):
                    patch_applied += 1
        except (json.JSONDecodeError, AttributeError):
            # Malformed report — count the instance but mark as unresolved.
            pass

    # Also try the top-level summary JSON produced by the harness.
    summary_candidates = list(eval_dir.rglob("*.json"))
    for sc in summary_candidates:
        if sc.name == "report.json":
            continue
        try:
            summary = json.loads(sc.read_text())
            if "total_instances" in summary:
                # Use the harness-provided totals when available.
                return {
                    "total_instances": summary.get("total_instances", total),
                    "resolved":        summary.get("resolved_instances", resolved),
                    "unresolved":      summary.get("unresolved_instances", total - resolved),
                    "patch_applied":   patch_applied,
                    "resolve_rate":    (
                        summary.get("resolved_instances", resolved)
                        / max(summary.get("total_instances", total), 1)
                    ),
                }
        except (json.JSONDecodeError, AttributeError):
            pass

    return {
        "total_instances": total,
        "resolved":        resolved,
        "unresolved":      total - resolved,
        "patch_applied":   patch_applied,
        "resolve_rate":    resolved / max(total, 1),
    }


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

        # Log all numeric metrics.
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        # Tag the run with the artifact location for easy retrieval.
        mlflow.set_tag("artifact_uri", artifact_uri)
        mlflow.set_tag("run_id", run_config["run_id"])

    print(
        f"[summarize_and_log] MLflow run logged to {tracking_uri} "
        f"under experiment '{experiment_name}'"
    )


def write_manifest(
    run_dir: Path,
    run_config: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Write ``manifest.json`` summarising the run artefacts.

    The manifest is the single document that lets a teammate reconstruct the
    full picture of a run: what was run, with which config, what was produced,
    and where to find each artefact.

    Args:
        run_dir:    Path to the run directory (``runs/<run-id>/``).
        run_config: Normalised run configuration dictionary.
        metrics:    Aggregated metrics dictionary.

    Returns:
        Path to the written ``manifest.json`` file.
    """
    preds_path = run_dir / "run-agent" / "preds.json"
    manifest = {
        "run_id":       run_config["run_id"],
        "created_at":   run_config["created_at"],
        "config":       str(run_dir / "config.json"),
        "artefacts": {
            "predictions":       str(preds_path) if preds_path.exists() else None,
            "trajectories_dir":  str(run_dir / "run-agent"),
            "eval_logs_dir":     str(run_dir / "run-eval"),
            "metrics":           str(run_dir / "metrics.json"),
        },
        "metrics_summary": metrics,
        "remote_storage": (
            "Not yet uploaded.  "
            "To upload: aws s3 cp --recursive "
            f"runs/{run_config['run_id']}/ "
            f"s3://<YOUR_BUCKET>/runs/{run_config['run_id']}/"
        ),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# ===========================================================================
# DAG definition
# ===========================================================================

@dag(
    dag_id="evaluate_agent",
    description=(
        "End-to-end SWE-bench evaluation: run mini-swe-agent on a configurable "
        "subset, evaluate with SWE-bench harness, and log results to MLflow."
    ),
    start_date=datetime(2024, 1, 1),
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
        """
        raw_params = context["dag_run"].conf or {}
        run_config = build_run_config(raw_params)
        run_dir = prepare_run_dir(run_config)
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

        Locates ``preds.json`` in the agent output directory and calls
        ``run_swebench_eval()``.

        Args:
            agent_xcom: Dictionary pushed by ``run_agent`` containing
                ``_run_dir`` and ``_agent_out_dir``.

        Returns:
            Dictionary with ``_eval_out_dir`` (str path) added for downstream
            tasks.

        Raises:
            FileNotFoundError: If ``preds.json`` does not exist in the agent
                output directory.
        """
        run_dir = Path(agent_xcom.pop("_run_dir"))
        agent_out_dir = Path(agent_xcom.pop("_agent_out_dir"))
        run_config = agent_xcom

        preds_path = agent_out_dir / "preds.json"
        if not preds_path.exists():
            raise FileNotFoundError(
                f"[run_eval] preds.json not found at {preds_path}. "
                "The run_agent task may have failed silently."
            )

        print(
            f"[run_eval] Starting SWE-bench evaluation\n"
            f"  predictions : {preds_path}\n"
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
