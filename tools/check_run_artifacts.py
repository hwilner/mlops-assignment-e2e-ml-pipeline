#!/usr/bin/env python3
"""Check that the run artefacts in this repository agree with each other.

Every run writes ``config.json``, ``metrics.json`` and ``manifest.json``.
The manifest is the document a teammate reads to understand a run, so it
must not disagree with the metrics it embeds, and the paths it lists must
resolve inside the run directory — an absolute path recorded inside a
container points nowhere for the person the directory is handed to.

The checker reports two levels of finding:

``ERROR``
    The artefacts contradict each other or claim something that is not
    there.  A run in this state cannot be reconstructed from its directory,
    so the checker exits non-zero.

``WARNING``
    The run is self-consistent but something is missing from the
    repository.  The most common case is a run whose evaluation reports
    were never committed, whose metrics therefore cannot be recomputed
    from the repository alone.

Usage:
    python tools/check_run_artifacts.py
    python tools/check_run_artifacts.py --runs-root runs

Example:
    $ python tools/check_run_artifacts.py
    WARNING run-20260706T205913-3e6f2910: no committed 'run-eval' directory,
        so the metrics cannot be recomputed from this repository
    0 errors, 1 warning
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The metrics statuses this checker understands.  A metrics file written
# before the status existed is accepted, and reported as a warning.
KNOWN_STATUSES = frozenset({"OK", "NO_EVALUATION_REPORTS", "INCONSISTENT_SOURCES"})

# The status that means the numbers in the metrics cannot be trusted.
INCONSISTENT_STATUS = "INCONSISTENT_SOURCES"

# Relative tolerance used when comparing the recorded resolve rate with the
# rate implied by the recorded counts.
_RATE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Finding:
    """One problem found in one run directory.

    Attributes:
        level: Either ``"ERROR"`` or ``"WARNING"``.
        run_id: The identifier of the run the finding belongs to.
        message: A description of the problem.
    """

    level: str
    run_id: str
    message: str

    def __str__(self) -> str:
        """Return the finding as a single line of text.

        Returns:
            The finding formatted as ``LEVEL run-id: message``.
        """
        return f"{self.level:<7} {self.run_id}: {self.message}"


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: The file to read.

    Returns:
        The parsed JSON value, or ``None`` if the file is missing or cannot
        be parsed.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _is_number(value: Any) -> bool:
    """Return whether *value* is a real number.

    Args:
        value: The value to test.

    Returns:
        ``True`` if *value* is an :class:`int` or a :class:`float` that is
        not a :class:`bool`.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_metrics(metrics: dict[str, Any], run_id: str) -> list[Finding]:
    """Check the internal consistency of one run's metrics.

    Args:
        metrics: The parsed ``metrics.json`` content.
        run_id: The identifier of the run, used in the findings.

    Returns:
        The findings for this metrics dictionary.
    """
    findings: list[Finding] = []

    total = metrics.get("total_instances")
    resolved = metrics.get("resolved")
    unresolved = metrics.get("unresolved")
    rate = metrics.get("resolve_rate")

    for name, value in (("total_instances", total), ("resolved", resolved),
                        ("unresolved", unresolved)):
        if not _is_number(value):
            findings.append(Finding("ERROR", run_id,
                                    f"metrics.{name} is missing or not a number"))
            return findings
        if value < 0:
            findings.append(Finding("ERROR", run_id,
                                    f"metrics.{name} is negative ({value})"))

    status = metrics.get("status")
    if status is None:
        findings.append(Finding(
            "WARNING", run_id,
            "metrics has no 'status' key, so it was written before the "
            "pipeline distinguished a measured result from an unmeasured one"))
    elif status not in KNOWN_STATUSES:
        findings.append(Finding("ERROR", run_id,
                                f"metrics.status is {status!r}, which is not "
                                f"one of {sorted(KNOWN_STATUSES)}"))
    elif status == INCONSISTENT_STATUS:
        findings.append(Finding(
            "ERROR", run_id,
            "metrics.status is INCONSISTENT_SOURCES, so the evaluation "
            "sources disagree and the numbers cannot be trusted"))

    if resolved + unresolved != total:
        findings.append(Finding(
            "ERROR", run_id,
            f"metrics counts do not add up: resolved ({resolved}) + "
            f"unresolved ({unresolved}) is not total_instances ({total})"))

    if total == 0:
        if rate is not None:
            findings.append(Finding(
                "ERROR", run_id,
                f"metrics.resolve_rate is {rate!r} but no instance was "
                "evaluated, so no rate is defined and the value must be null"))
    elif rate is None:
        findings.append(Finding(
            "ERROR", run_id,
            "metrics.resolve_rate is null although "
            f"{total} instance(s) were evaluated"))
    elif not _is_number(rate):
        findings.append(Finding("ERROR", run_id,
                                f"metrics.resolve_rate is not a number ({rate!r})"))
    else:
        expected = resolved / total
        if abs(rate - expected) > _RATE_TOLERANCE:
            findings.append(Finding(
                "ERROR", run_id,
                f"metrics.resolve_rate is {rate!r} but the recorded counts "
                f"imply {expected!r}"))

    return findings


def check_manifest(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    run_dir: Path,
    run_id: str,
) -> list[Finding]:
    """Check that one run's manifest agrees with its metrics and its files.

    Args:
        manifest: The parsed ``manifest.json`` content.
        metrics: The parsed ``metrics.json`` content.
        run_dir: The run directory the manifest belongs to.
        run_id: The identifier of the run, used in the findings.

    Returns:
        The findings for this manifest.
    """
    findings: list[Finding] = []

    if manifest.get("metrics_summary") != metrics:
        findings.append(Finding(
            "ERROR", run_id,
            "manifest.metrics_summary does not match metrics.json, so the "
            "run reports two different results"))

    manifest_run_id = manifest.get("run_id")
    if manifest_run_id != run_id:
        findings.append(Finding(
            "ERROR", run_id,
            f"manifest.run_id is {manifest_run_id!r}, which does not match "
            f"the run directory name {run_id!r}"))

    artefacts = manifest.get("artefacts")
    if not isinstance(artefacts, dict):
        findings.append(Finding("ERROR", run_id,
                                "manifest.artefacts is missing or not an object"))
        return findings

    for name, value in artefacts.items():
        if value is None:
            # A null artefact means the file was not produced, which the
            # manifest is allowed to record.
            continue
        if not isinstance(value, str):
            findings.append(Finding("ERROR", run_id,
                                    f"manifest artefact {name!r} is not a string"))
            continue
        if Path(value).is_absolute():
            findings.append(Finding(
                "ERROR", run_id,
                f"manifest artefact {name!r} is the absolute path {value!r}, "
                "which does not resolve for anyone outside the machine that "
                "produced the run"))
            continue
        resolved = (run_dir / value).resolve()
        if not str(resolved).startswith(str(run_dir.resolve()) + "/"):
            findings.append(Finding(
                "ERROR", run_id,
                f"manifest artefact {name!r} points outside the run "
                f"directory ({value!r})"))
            continue
        if not resolved.exists():
            findings.append(Finding(
                "ERROR", run_id,
                f"manifest artefact {name!r} points at {value!r}, which does "
                "not exist"))

    return findings


def check_run(run_dir: Path) -> list[Finding]:
    """Check one run directory.

    Args:
        run_dir: The run directory to check.

    Returns:
        The findings for this run directory.
    """
    run_id = run_dir.name
    findings: list[Finding] = []

    config = _read_json(run_dir / "config.json")
    if not isinstance(config, dict):
        findings.append(Finding("ERROR", run_id,
                                "config.json is missing or is not a JSON object"))
    else:
        config_run_id = config.get("run_id")
        if config_run_id != run_id:
            findings.append(Finding(
                "ERROR", run_id,
                f"config.json names the run {config_run_id!r}, which does "
                f"not match the run directory name {run_id!r}"))

    metrics = _read_json(run_dir / "metrics.json")
    if not isinstance(metrics, dict):
        findings.append(Finding("ERROR", run_id,
                                "metrics.json is missing or is not a JSON object"))
        metrics = None

    manifest = _read_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        findings.append(Finding("ERROR", run_id,
                                "manifest.json is missing or is not a JSON object"))
        manifest = None

    if metrics is not None:
        findings.extend(check_metrics(metrics, run_id))
    if manifest is not None and metrics is not None:
        findings.extend(check_manifest(manifest, metrics, run_dir, run_id))

    eval_dir = run_dir / "run-eval"
    if not eval_dir.is_dir():
        findings.append(Finding(
            "WARNING", run_id,
            "no committed 'run-eval' directory, so the metrics cannot be "
            "recomputed from this repository"))
    elif not any(eval_dir.rglob("report.json")):
        findings.append(Finding(
            "WARNING", run_id,
            "the committed 'run-eval' directory holds no instance report, "
            "so the metrics cannot be recomputed from this repository"))

    return findings


def check_runs(runs_root: Path) -> list[Finding]:
    """Check every run directory under *runs_root*.

    Args:
        runs_root: The directory that holds the runs.

    Returns:
        The findings for every run directory, in directory-name order.
    """
    findings: list[Finding] = []
    if not runs_root.is_dir():
        return findings
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        findings.extend(check_run(run_dir))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the checker.

    Args:
        argv: The command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when no error was found, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs",
        help="Directory that holds the runs (default: the repository's 'runs' directory).",
    )
    arguments = parser.parse_args(argv)

    findings = check_runs(arguments.runs_root)
    for finding in findings:
        print(finding)

    errors = sum(1 for finding in findings if finding.level == "ERROR")
    warnings = sum(1 for finding in findings if finding.level == "WARNING")
    print(f"{errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
