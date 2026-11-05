"""Tests for the run-artefact checker in ``tools/check_run_artifacts.py``.

Every check the checker performs is exercised here against a run directory
that violates it, so a check that cannot fail cannot be merged.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools import check_run_artifacts as checker

METRICS = {
    "status": "OK",
    "total_instances": 3,
    "resolved": 1,
    "unresolved": 2,
    "patch_applied": 3,
    "resolve_rate": 1 / 3,
    "dataset_instances": 500,
    "submitted_instances": 3,
    "malformed_reports": 0,
    "notes": [],
}


def write_json(path: Path, payload: object) -> None:
    """Write *payload* as JSON to *path*.

    Args:
        path: The file to write.
        payload: The JSON-serialisable payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def make_run(runs_root: Path, run_id: str = "run-1") -> Path:
    """Create a run directory whose artefacts agree with each other.

    Args:
        runs_root: The directory that holds the runs.
        run_id: The identifier of the run.

    Returns:
        The path of the run directory.
    """
    run_dir = runs_root / run_id
    write_json(run_dir / "config.json", {"run_id": run_id, "split": "test"})
    write_json(run_dir / "metrics.json", METRICS)
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_at": "2026-07-06T20:59:13.383363Z",
            "config": "config.json",
            "artefacts": {
                "predictions": "run-agent/preds.json",
                "trajectories_dir": "run-agent",
                "eval_logs_dir": "run-eval",
                "metrics": "metrics.json",
            },
            "metrics_summary": METRICS,
        },
    )
    write_json(run_dir / "run-agent" / "preds.json", {"inst-0": {}})
    write_json(
        run_dir / "run-eval" / "inst-0" / "report.json",
        {"inst-0": {"resolved": True, "patch_successfully_applied": True}},
    )
    return run_dir


def errors_for(run_dir: Path) -> list[str]:
    """Return the error messages the checker reports for one run.

    Args:
        run_dir: The run directory to check.

    Returns:
        The messages of the findings that are errors.
    """
    return [
        finding.message
        for finding in checker.check_run(run_dir)
        if finding.level == "ERROR"
    ]


def warnings_for(run_dir: Path) -> list[str]:
    """Return the warning messages the checker reports for one run.

    Args:
        run_dir: The run directory to check.

    Returns:
        The messages of the findings that are warnings.
    """
    return [
        finding.message
        for finding in checker.check_run(run_dir)
        if finding.level == "WARNING"
    ]


class TestConsistentRun:
    """A run whose artefacts agree is accepted."""

    def test_reports_no_error(self, tmp_path: Path) -> None:
        """A self-consistent run produces no error."""
        run_dir = make_run(tmp_path)
        assert errors_for(run_dir) == []

    def test_the_command_line_exits_zero(self, tmp_path: Path) -> None:
        """The checker exits zero when nothing is wrong."""
        make_run(tmp_path)
        assert checker.main(["--runs-root", str(tmp_path)]) == 0


class TestViolationsAreCaught:
    """Each check is exercised against a run that violates it."""

    def test_a_manifest_that_contradicts_the_metrics(self, tmp_path: Path) -> None:
        """The manifest must not report a different result."""
        run_dir = make_run(tmp_path)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metrics_summary"] = {**METRICS, "total_instances": 0,
                                       "resolved": 0, "unresolved": 0,
                                       "resolve_rate": 0.0}
        write_json(manifest_path, manifest)

        assert any(
            "does not match metrics.json" in message
            for message in errors_for(run_dir)
        )

    def test_an_absolute_artefact_path(self, tmp_path: Path) -> None:
        """An absolute path does not resolve for the recipient."""
        run_dir = make_run(tmp_path)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artefacts"]["metrics"] = "/opt/airflow/runs/run-1/metrics.json"
        write_json(manifest_path, manifest)

        assert any(
            "absolute path" in message for message in errors_for(run_dir)
        )

    def test_an_artefact_that_does_not_exist(self, tmp_path: Path) -> None:
        """A path that resolves to nothing is an error."""
        run_dir = make_run(tmp_path)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artefacts"]["predictions"] = "run-agent/preds.json"
        (run_dir / "run-agent" / "preds.json").unlink()
        write_json(manifest_path, manifest)

        assert any(
            "does not exist" in message for message in errors_for(run_dir)
        )

    def test_an_artefact_outside_the_run_directory(self, tmp_path: Path) -> None:
        """A path that leaves the run directory is an error."""
        run_dir = make_run(tmp_path)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artefacts"]["predictions"] = "../other-run/preds.json"
        write_json(manifest_path, manifest)

        assert any(
            "outside the run directory" in message
            for message in errors_for(run_dir)
        )

    def test_a_rate_for_a_run_that_evaluated_nothing(
        self, tmp_path: Path
    ) -> None:
        """A resolve rate needs a denominator."""
        run_dir = make_run(tmp_path)
        metrics = {**METRICS, "total_instances": 0, "resolved": 0,
                   "unresolved": 0, "resolve_rate": 0.0}
        write_json(run_dir / "metrics.json", metrics)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metrics_summary"] = metrics
        write_json(manifest_path, manifest)

        assert any(
            "no rate is defined" in message for message in errors_for(run_dir)
        )

    def test_counts_that_do_not_add_up(self, tmp_path: Path) -> None:
        """Resolved plus unresolved must be the total."""
        run_dir = make_run(tmp_path)
        metrics = {**METRICS, "unresolved": 0}
        write_json(run_dir / "metrics.json", metrics)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metrics_summary"] = metrics
        write_json(manifest_path, manifest)

        assert any(
            "do not add up" in message for message in errors_for(run_dir)
        )

    def test_a_rate_that_does_not_match_the_counts(self, tmp_path: Path) -> None:
        """The recorded rate must be the rate of the recorded counts."""
        run_dir = make_run(tmp_path)
        metrics = {**METRICS, "resolve_rate": 0.0}
        write_json(run_dir / "metrics.json", metrics)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metrics_summary"] = metrics
        write_json(manifest_path, manifest)

        assert any(
            "the recorded counts imply" in message
            for message in errors_for(run_dir)
        )

    def test_a_run_id_that_does_not_match_the_directory(
        self, tmp_path: Path
    ) -> None:
        """The run identifier must be the same everywhere."""
        run_dir = make_run(tmp_path)
        write_json(run_dir / "config.json", {"run_id": "run-other"})

        assert any(
            "does not match the run directory name" in message
            for message in errors_for(run_dir)
        )

    def test_inconsistent_sources(self, tmp_path: Path) -> None:
        """A run whose sources disagree is not trustworthy."""
        run_dir = make_run(tmp_path)
        metrics = {**METRICS, "status": "INCONSISTENT_SOURCES"}
        write_json(run_dir / "metrics.json", metrics)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metrics_summary"] = metrics
        write_json(manifest_path, manifest)

        assert any(
            "cannot be trusted" in message for message in errors_for(run_dir)
        )

    def test_a_missing_metrics_file(self, tmp_path: Path) -> None:
        """A run without metrics cannot be reconstructed."""
        run_dir = make_run(tmp_path)
        (run_dir / "metrics.json").unlink()

        assert any(
            "metrics.json is missing" in message for message in errors_for(run_dir)
        )

    def test_the_command_line_exits_non_zero(self, tmp_path: Path) -> None:
        """The checker exits non-zero when an error is found."""
        run_dir = make_run(tmp_path)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artefacts"]["metrics"] = "/opt/airflow/runs/run-1/metrics.json"
        write_json(manifest_path, manifest)

        assert checker.main(["--runs-root", str(tmp_path)]) == 1


class TestWarnings:
    """A missing artefact is reported without failing the check."""

    def test_a_run_without_evaluation_reports(self, tmp_path: Path) -> None:
        """The metrics cannot be recomputed without the reports."""
        run_dir = make_run(tmp_path)
        for report in (run_dir / "run-eval").rglob("report.json"):
            report.unlink()

        assert errors_for(run_dir) == []
        assert any(
            "cannot be recomputed" in message
            for message in warnings_for(run_dir)
        )

    def test_metrics_without_a_status(self, tmp_path: Path) -> None:
        """A metrics file written before the status existed is a warning."""
        run_dir = make_run(tmp_path)
        legacy = {key: value for key, value in METRICS.items() if key != "status"}
        write_json(run_dir / "metrics.json", legacy)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metrics_summary"] = legacy
        write_json(manifest_path, manifest)

        assert errors_for(run_dir) == []
        assert any(
            "no 'status' key" in message for message in warnings_for(run_dir)
        )
