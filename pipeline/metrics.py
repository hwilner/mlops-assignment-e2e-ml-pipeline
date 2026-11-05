"""Aggregation of the SWE-bench evaluation output into run metrics.

The evaluation harness leaves two kinds of output in its report directory:
one ``report.json`` per evaluated instance, and one aggregate summary file.
This module turns that output into the metrics dictionary that is written to
``metrics.json``, embedded in ``manifest.json`` and logged to MLflow.

Two properties of those metrics matter more than the numbers themselves.

**A run that evaluated nothing must not look like a run that resolved
nothing.**  The harness reports ``total_instances`` as the number of
instances in the *dataset*, which for SWE-bench Verified is 500, while the
number of instances that were actually evaluated is whatever subset was run.
Dividing the resolved count by the dataset size understates the resolve rate
by the ratio between the two, and a run in which the harness produced no
report at all would be reported as a resolve rate of 0.0 — indistinguishable
from a run in which the agent tried and failed.  The metrics therefore count
the *evaluated* instances and report the dataset size separately, and they
carry a ``status`` that says whether anything was measured at all.

**The counts must add up.**  Every count in the metrics dictionary describes
the same set of instances, so ``resolved + unresolved`` is always
``total_instances``, and ``resolve_rate`` is always ``resolved /
total_instances`` — or ``None`` when nothing was evaluated, because no rate
is defined for an empty set.

Example:
    >>> import json, tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     metrics = collect_metrics(Path(directory))
    ...     metrics["status"], metrics["resolve_rate"]
    ('NO_EVALUATION_REPORTS', None)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Every instance was evaluated and the counts are consistent.
METRICS_STATUS_OK = "OK"

# The evaluation produced no per-instance report, so no result was measured.
METRICS_STATUS_NO_REPORTS = "NO_EVALUATION_REPORTS"

# The harness summary and the per-instance reports disagree, so neither can
# be trusted on its own.
METRICS_STATUS_INCONSISTENT = "INCONSISTENT_SOURCES"

# The keys of the metrics dictionary, in the order they are written.
METRICS_KEYS: tuple[str, ...] = (
    "status",
    "total_instances",
    "resolved",
    "unresolved",
    "patch_applied",
    "resolve_rate",
    "dataset_instances",
    "submitted_instances",
    "malformed_reports",
    "notes",
)


def _read_report(report_path: Path) -> tuple[bool, bool] | None:
    """Read the resolved and patch-applied flags from one instance report.

    Args:
        report_path: Path of a per-instance ``report.json`` file.

    Returns:
        A tuple ``(resolved, patch_applied)``, or ``None`` if the file could
        not be read as an instance report.
    """
    try:
        payload = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    resolved = False
    patch_applied = False
    for instance_data in payload.values():
        if not isinstance(instance_data, dict):
            return None
        resolved = resolved or bool(instance_data.get("resolved"))
        patch_applied = patch_applied or bool(
            instance_data.get("patch_successfully_applied")
        )
    return resolved, patch_applied


def _find_harness_summary(eval_dir: Path) -> dict[str, Any] | None:
    """Return the harness summary found in *eval_dir*, if there is one.

    The SWE-bench harness writes one aggregate file next to the per-instance
    reports.  It is recognised by the ``total_instances`` key it carries.

    Args:
        eval_dir: The ``run-eval/`` directory of a run.

    Returns:
        The parsed summary dictionary, or ``None`` if the directory holds no
        summary.
    """
    for candidate in sorted(eval_dir.rglob("*.json")):
        if candidate.name == "report.json":
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "total_instances" in payload:
            return payload
    return None


def collect_metrics(eval_dir: Path) -> dict[str, Any]:
    """Aggregate the evaluation output of one run into a metrics dictionary.

    Args:
        eval_dir: The ``run-eval/`` directory produced by the evaluation
            stage.

    Returns:
        A dictionary with the keys listed in :data:`METRICS_KEYS`.  The
        ``status`` is one of :data:`METRICS_STATUS_OK`,
        :data:`METRICS_STATUS_NO_REPORTS` or
        :data:`METRICS_STATUS_INCONSISTENT`.  ``resolve_rate`` is ``None``
        when no instance was evaluated, because no rate is defined for an
        empty set.  ``total_instances`` counts the instances that were
        evaluated; the size of the dataset the harness loaded is reported
        separately as ``dataset_instances``.
    """
    report_files = sorted(eval_dir.rglob("report.json"))

    evaluated = 0
    resolved = 0
    patch_applied = 0
    malformed_reports = 0
    notes: list[str] = []

    for report_path in report_files:
        flags = _read_report(report_path)
        if flags is None:
            malformed_reports += 1
            # The instance was evaluated but its outcome cannot be read, so
            # it counts as unresolved rather than being dropped.
            evaluated += 1
            notes.append(f"Unreadable instance report: {report_path.name}")
            continue
        instance_resolved, instance_patch_applied = flags
        evaluated += 1
        resolved += int(instance_resolved)
        patch_applied += int(instance_patch_applied)

    summary = _find_harness_summary(eval_dir)
    dataset_instances = None
    submitted_instances = None
    status = METRICS_STATUS_OK

    if summary is not None:
        dataset_instances = summary.get("total_instances")
        submitted_instances = summary.get("submitted_instances")
        summary_resolved = summary.get("resolved_instances")
        if (
            summary_resolved is not None
            and evaluated > 0
            and int(summary_resolved) != resolved
        ):
            # The two sources count the same thing and disagree, so the
            # run is reported as inconsistent instead of one of them being
            # preferred silently.
            status = METRICS_STATUS_INCONSISTENT
            notes.append(
                f"The harness summary reports {summary_resolved} resolved "
                f"instances but {resolved} instance reports do."
            )
        if submitted_instances is not None and evaluated == 0:
            notes.append(
                f"The harness received {submitted_instances} predictions but "
                "produced no per-instance report, so no instance was "
                "evaluated."
            )
    elif evaluated == 0:
        notes.append(
            "The evaluation directory holds no instance report and no "
            "harness summary, so no instance was evaluated."
        )

    if evaluated == 0 and status == METRICS_STATUS_OK:
        status = METRICS_STATUS_NO_REPORTS

    return {
        "status": status,
        "total_instances": evaluated,
        "resolved": resolved,
        "unresolved": evaluated - resolved,
        "patch_applied": patch_applied,
        "resolve_rate": (resolved / evaluated) if evaluated > 0 else None,
        "dataset_instances": dataset_instances,
        "submitted_instances": submitted_instances,
        "malformed_reports": malformed_reports,
        "notes": notes,
    }


def write_manifest(
    run_dir: Path,
    run_config: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Write ``manifest.json`` describing the artefacts of one run.

    The manifest is the document that lets a teammate reconstruct a run, so
    every path in it is relative to the run directory: an absolute path
    recorded inside a container is meaningless to the person the directory
    is handed to.  An artefact that was not produced is recorded as ``null``
    rather than as a path that does not resolve.

    Args:
        run_dir: Path of the run directory (``runs/<run-id>/``).
        run_config: Normalised run configuration.
        metrics: The metrics dictionary, embedded verbatim so that the
            manifest and ``metrics.json`` cannot disagree.

    Returns:
        The path of the written ``manifest.json``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def relative_artefact(*parts: str) -> str | None:
        """Return the run-relative path of an artefact, or None if absent."""
        candidate = run_dir.joinpath(*parts)
        return "/".join(parts) if candidate.exists() else None

    manifest = {
        "run_id": run_config["run_id"],
        "created_at": run_config.get("created_at"),
        "config": relative_artefact("config.json"),
        "artefacts": {
            "predictions": relative_artefact("run-agent", "preds.json"),
            "trajectories_dir": relative_artefact("run-agent"),
            "eval_logs_dir": relative_artefact("run-eval"),
            "metrics": relative_artefact("metrics.json"),
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
