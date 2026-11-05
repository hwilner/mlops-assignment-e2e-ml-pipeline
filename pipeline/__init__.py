"""Airflow-free pipeline logic for the ``evaluate_agent`` DAG.

The DAG in ``dags/evaluate_agent.py`` is deliberately thin: it wires tasks
together and talks to Airflow and MLflow.  Everything that can be decided
without a scheduler, a Docker daemon or an inference API key lives in this
package, so that it can be unit-tested in CI and reasoned about on its own.

The package is split by the question each module answers:

* :mod:`pipeline.config` — what run is this?  Builds and validates the run
  configuration that every other stage is driven by.
* :mod:`pipeline.evaluation` — what commands run the agent and the
  evaluation?  Builds the two command lines as pure data.
* :mod:`pipeline.metrics` — what did the evaluation actually measure?
  Aggregates the harness output into metrics that distinguish a measured
  result from a result that was never computed.
"""

from pipeline.config import (
    DEFAULT_COST_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_TASK_SLICE,
    DEFAULT_WORKERS,
    build_run_config,
    prepare_run_dir,
    validate_run_id,
)
from pipeline.evaluation import (
    build_agent_command,
    build_eval_command,
    resolve_dataset_name,
)
from pipeline.metrics import (
    METRICS_STATUS_INCONSISTENT,
    METRICS_STATUS_NO_REPORTS,
    METRICS_STATUS_OK,
    collect_metrics,
    write_manifest,
)
from pipeline.predictions import read_predictions

__all__ = [
    "DEFAULT_COST_LIMIT",
    "DEFAULT_MODEL",
    "DEFAULT_TASK_SLICE",
    "DEFAULT_WORKERS",
    "METRICS_STATUS_INCONSISTENT",
    "METRICS_STATUS_NO_REPORTS",
    "METRICS_STATUS_OK",
    "build_agent_command",
    "build_eval_command",
    "build_run_config",
    "collect_metrics",
    "prepare_run_dir",
    "read_predictions",
    "resolve_dataset_name",
    "validate_run_id",
    "write_manifest",
]
