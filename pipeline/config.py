"""Run-configuration handling for the ``evaluate_agent`` Airflow DAG.

The run configuration is the contract between the Airflow trigger parameters
and everything the pipeline writes: the run directory, the agent command
line, the evaluation command line, ``metrics.json`` and ``manifest.json``.
Keeping that contract in a module that does not import Airflow makes it
testable without a scheduler, a Docker daemon or an inference API key.

Every parameter is validated here rather than at the point of use.  A
parameter that is rejected during configuration fails with a message naming
the parameter and the accepted range, whereas a parameter that is only
noticed by a subprocess fails with that subprocess's error message, which
usually says nothing about the pipeline.

Example:
    >>> from pipeline.config import build_run_config
    >>> config = build_run_config({"split": "test", "subset": "verified", "workers": 3})
    >>> config["split"], config["workers"]
    ('test', 3)
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"
DEFAULT_TASK_SLICE = "0:3"
DEFAULT_WORKERS = 3
DEFAULT_COST_LIMIT = 0

# The run identifier becomes a directory name under the runs root and is also
# used as the MLflow run name, so it is restricted to characters that are
# safe in both places.  The leading-character rule keeps a run identifier
# from starting with a dot (which would hide the run directory) or a dash.
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# The longest run identifier accepted.  The bound keeps a run identifier from
# exceeding the file-name limits of the file systems the runs root may live on.
_MAX_RUN_ID_LENGTH = 128


def _now_utc() -> datetime:
    """Return the current UTC time.

    Returns:
        The current time as a timezone-aware :class:`~datetime.datetime`
        in UTC.
    """
    return datetime.now(UTC)


def _format_timestamp(moment: datetime) -> str:
    """Format *moment* the way the run artefacts record timestamps.

    Args:
        moment: A timezone-aware :class:`~datetime.datetime`.

    Returns:
        An ISO-8601 timestamp with a trailing ``Z``, for example
        ``2026-07-06T20:59:13.383363Z``.
    """
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def validate_run_id(run_id: Any) -> str:
    """Validate that *run_id* is safe to use as a run-directory name.

    The run identifier is supplied by the person triggering the DAG and is
    used as a path component under the runs root.  Without validation a
    value such as ``../../etc`` or ``/tmp/other`` would place the run
    directory outside the runs root, so the whole run — including its
    ``metrics.json`` and ``manifest.json`` — would be written somewhere the
    pipeline does not own.

    Args:
        run_id: The candidate run identifier.

    Returns:
        The validated run identifier, unchanged.

    Raises:
        ValueError: If *run_id* is not a non-empty string of at most 128
            characters that begins with an alphanumeric character and
            contains only letters, digits, dots, underscores and dashes.
    """
    if not isinstance(run_id, str):
        raise ValueError(
            f"Run id must be a string, got {type(run_id).__name__}. "
            "Trigger the DAG with a run_id such as 'run-2026-07-06-01'."
        )
    if not run_id.strip():
        raise ValueError("Run id must not be empty or whitespace only.")
    if len(run_id) > _MAX_RUN_ID_LENGTH:
        raise ValueError(
            f"Run id must be at most {_MAX_RUN_ID_LENGTH} characters, "
            f"got {len(run_id)}."
        )
    if not _RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"Run id {run_id!r} is not a valid run directory name. It must "
            "start with a letter or digit and may contain only letters, "
            "digits, dots, underscores and dashes. Path separators, '..' "
            "and absolute paths are rejected because the run id is used as "
            "a directory name under the runs root."
        )
    return run_id


def _validate_workers(value: Any) -> int:
    """Validate the ``workers`` parameter.

    Args:
        value: The raw value supplied by the caller.

    Returns:
        The number of parallel workers as an :class:`int`.

    Raises:
        ValueError: If *value* is not an integer of at least 1.  Zero
            workers is rejected because both the agent runner and the
            evaluation harness would then process no instance at all, which
            would look like a completed run with nothing resolved.
    """
    if isinstance(value, bool):
        raise ValueError("Param 'workers' must be an integer, got a boolean.")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Param 'workers' must be a whole number, got {value!r}."
            )
        value = int(value)
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            raise ValueError(
                f"Param 'workers' must be an integer, got {value!r}."
            ) from None
    if not isinstance(value, int):
        raise ValueError(
            f"Param 'workers' must be an integer, got {type(value).__name__}."
        )
    if value < 1:
        raise ValueError(
            f"Param 'workers' must be at least 1, got {value}. "
            "With zero workers no instance is evaluated, and the run would "
            "be reported as a run that resolved nothing."
        )
    return value


def _validate_cost_limit(value: Any) -> float:
    """Validate the ``cost_limit`` parameter.

    Args:
        value: The raw value supplied by the caller.

    Returns:
        The per-task cost limit in US dollars as a :class:`float`.

    Raises:
        ValueError: If *value* is not a number of at least 0.
    """
    if isinstance(value, bool):
        raise ValueError("Param 'cost_limit' must be a number, got a boolean.")
    try:
        cost_limit = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Param 'cost_limit' must be a number, got {value!r}."
        ) from None
    if cost_limit < 0:
        raise ValueError(
            f"Param 'cost_limit' must not be negative, got {cost_limit}. "
            "Use 0 to mean 'no limit'."
        )
    return cost_limit


def _validate_text_param(name: str, value: Any) -> str:
    """Validate a required, non-empty text parameter.

    Args:
        name: The parameter name, used in the error message.
        value: The raw value supplied by the caller.

    Returns:
        The value as a stripped :class:`str`.

    Raises:
        ValueError: If *value* is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Param {name!r} must be a non-empty string, got {value!r}."
        )
    return value.strip()


def build_run_config(params: Mapping[str, Any]) -> dict[str, Any]:
    """Build a normalised run configuration from raw Airflow parameters.

    Applies the defaults for the optional parameters, validates every
    parameter, and generates a run identifier when one was not supplied.
    The returned dictionary is the single source of truth for the entire DAG
    run and is written to ``config.json`` by ``prepare_run_dir``.

    Args:
        params: Raw Airflow ``dag_run.conf`` mapping.  Must contain at
            minimum ``split``, ``subset`` and ``workers``.

    Returns:
        A fully populated run-configuration dictionary with the keys
        ``run_id``, ``split``, ``subset``, ``workers``, ``model``,
        ``task_slice``, ``cost_limit`` and ``created_at``.

    Raises:
        ValueError: If a required parameter is missing, or if any parameter
            fails validation.

    Example:
        >>> config = build_run_config(
        ...     {"split": "test", "subset": "verified", "workers": 3}
        ... )
        >>> config["task_slice"], config["cost_limit"]
        ('0:3', 0)
    """
    for required_key in ("split", "subset", "workers"):
        if required_key not in params:
            raise ValueError(
                f"Required Airflow param '{required_key}' is missing. "
                "Trigger the DAG with at least: "
                '{"split": "test", "subset": "verified", "workers": 3}'
            )

    supplied_run_id = params.get("run_id")
    if supplied_run_id:
        run_id = validate_run_id(supplied_run_id)
    else:
        run_id = f"run-{_now_utc().strftime('%Y%m%dT%H%M%S')}-{str(uuid.uuid4())[:8]}"

    model = params.get("model", DEFAULT_MODEL)
    task_slice = params.get("task_slice", DEFAULT_TASK_SLICE)

    return {
        "run_id": run_id,
        "split": _validate_text_param("split", params["split"]),
        "subset": _validate_text_param("subset", params["subset"]),
        "workers": _validate_workers(params["workers"]),
        "model": _validate_text_param("model", model),
        "task_slice": _validate_text_param("task_slice", task_slice),
        "cost_limit": _validate_cost_limit(params.get("cost_limit", DEFAULT_COST_LIMIT)),
        "created_at": _format_timestamp(_now_utc()),
    }


def prepare_run_dir(run_config: Mapping[str, Any], runs_root: Path) -> Path:
    """Create the run directory tree and write ``config.json``.

    Creates the following layout under ``<runs_root>/<run-id>/``::

        <run-id>/
            config.json      ← the full run configuration
            run-agent/       ← populated by the agent stage
            run-eval/        ← populated by the evaluation stage

    Args:
        run_config: Normalised run configuration as built by
            :func:`build_run_config`.
        runs_root: The directory that holds all runs.

    Returns:
        The path of the run directory.
    """
    run_dir = Path(runs_root) / str(run_config["run_id"])
    (run_dir / "run-agent").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dict(run_config), indent=2))
    return run_dir
