"""Command construction for the agent and evaluation stages.

Both stages of the pipeline shell out to a command-line tool whose interface
is owned by another project.  Building those command lines as pure data —
rather than inline in the DAG — keeps the contract with ``mini-swe-agent``
and ``SWE-bench`` in one readable place and makes it testable without an
inference API key, a Docker daemon or a SWE-bench image.

Example:
    >>> from pipeline.evaluation import build_eval_command
    >>> config = {
    ...     "run_id": "run-1",
    ...     "split": "test",
    ...     "subset": "verified",
    ...     "workers": 3,
    ... }
    >>> build_eval_command(config, "preds.json", "run-eval")[-4:]
    ['--run_id', 'run-1', '--report_dir', 'run-eval']
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The SWE-bench dataset that corresponds to each dataset subset the pipeline
# accepts.  A subset that is not listed here is passed through unchanged, so
# a fully-qualified dataset name can be supplied directly.
DATASET_BY_SUBSET: dict[str, str] = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}

def resolve_dataset_name(subset: str) -> str:
    """Return the SWE-bench dataset name that corresponds to *subset*.

    Args:
        subset: The dataset subset, for example ``"verified"``.

    Returns:
        The dataset name to hand to the SWE-bench evaluation harness.  An
        unrecognised subset is returned unchanged, so callers can supply a
        fully-qualified dataset name directly.

    Example:
        >>> resolve_dataset_name("lite")
        'princeton-nlp/SWE-bench_Lite'
        >>> resolve_dataset_name("my-org/my-dataset")
        'my-org/my-dataset'
    """
    return DATASET_BY_SUBSET.get(subset.lower(), subset)


def build_agent_command(
    run_config: Mapping[str, Any],
    output_dir: Path | str,
    package_root: Path | None = None,
) -> list[str]:
    """Build the ``mini-extra swebench`` command that runs the agent.

    The per-task cost limit is applied through the agent configuration
    rather than through a command-line option: the batch subcommand of
    ``mini-extra`` has no ``--cost-limit`` option (only the single-instance
    subcommand does), and passing it makes mini-swe-agent exit with
    ``No such option: --cost-limit`` before any instance is attempted.

    The limit is always passed, including when it is zero.  Zero means "no
    limit" in this pipeline, but the benchmark configuration that
    mini-swe-agent ships with caps each task at 3 US dollars, so leaving
    the limit unset would apply a cap that nobody asked for.

    Args:
        run_config: Normalised run configuration as built by
            :func:`pipeline.config.build_run_config`.
        output_dir: Directory the agent writes its predictions and
            trajectories to.
        package_root: Root of the installed ``minisweagent`` package.  Only
            supplied by the tests.

    Returns:
        The argument vector to execute.

    Raises:
        FileNotFoundError: If the mini-swe-agent benchmark configuration
            cannot be located.
    """
    command = [
        "uv",
        "run",
        "mini-extra",
        "swebench",
        "--subset",
        str(run_config["subset"]),
        "--split",
        str(run_config["split"]),
        "--model",
        str(run_config["model"]),
        "--slice",
        str(run_config["task_slice"]),
        "--workers",
        str(run_config["workers"]),
        "-o",
        str(output_dir),
    ]
    if float(run_config["cost_limit"]) > 0:
        command += ["--cost-limit", str(run_config["cost_limit"])]
    return command


def build_eval_command(
    run_config: Mapping[str, Any],
    preds_path: Path | str,
    report_dir: Path | str,
) -> list[str]:
    """Build the SWE-bench evaluation harness command.

    The split is passed to the harness as well as to the agent.  Without it
    the harness falls back to its own default split, so predictions for a
    different split would never match an instance of the dataset and the
    harness would produce no report at all.

    Args:
        run_config: Normalised run configuration as built by
            :func:`pipeline.config.build_run_config`.
        preds_path: Path of the ``preds.json`` produced by the agent.
        report_dir: Directory the harness writes its reports to.

    Returns:
        The argument vector to execute.
    """
    return [
        "uv",
        "run",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        resolve_dataset_name(str(run_config["subset"])),
        "--predictions_path",
        str(preds_path),
        "--split",
        str(run_config["split"]),
        "--max_workers",
        str(run_config["workers"]),
        "--run_id",
        str(run_config["run_id"]),
        "--report_dir",
        str(report_dir),
    ]
