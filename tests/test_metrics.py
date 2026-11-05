"""Tests for the metric aggregation in :mod:`pipeline.metrics`."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.metrics import (
    METRICS_STATUS_INCONSISTENT,
    METRICS_STATUS_NO_REPORTS,
    METRICS_STATUS_OK,
    collect_metrics,
    write_manifest,
)


def write_report(directory: Path, instance_id: str, **flags: object) -> Path:
    """Write one per-instance report the way the harness writes it.

    Args:
        directory: The directory to write the report into.
        instance_id: The identifier of the instance the report belongs to.
        **flags: The report fields, such as ``resolved``.

    Returns:
        The path of the written report.
    """
    instance_dir = directory / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    report_path = instance_dir / "report.json"
    report_path.write_text(json.dumps({instance_id: dict(flags)}))
    return report_path


def write_summary(directory: Path, **fields: object) -> Path:
    """Write a harness summary the way the harness writes it.

    Args:
        directory: The directory to write the summary into.
        **fields: The summary fields, such as ``total_instances``.

    Returns:
        The path of the written summary.
    """
    summary_path = directory / "nebius__moonshotai__Kimi-K2.6.run-1.json"
    summary_path.write_text(json.dumps(dict(fields)))
    return summary_path


class TestCollectMetrics:
    """The metrics must describe what was evaluated, not what exists."""

    def test_an_unevaluated_run_reports_no_rate(self, tmp_path: Path) -> None:
        """A run that evaluated nothing must not report a resolve rate.

        A rate of 0.0 would be indistinguishable from a run in which the
        agent tried every instance and resolved none of them.
        """
        metrics = collect_metrics(tmp_path)

        assert metrics["status"] == METRICS_STATUS_NO_REPORTS
        assert metrics["total_instances"] == 0
        assert metrics["resolve_rate"] is None
        assert metrics["notes"]

    def test_the_rate_is_over_the_evaluated_instances(
        self, tmp_path: Path
    ) -> None:
        """The denominator is the number of instances evaluated.

        The harness reports ``total_instances`` as the size of the dataset,
        which is 500 for SWE-bench Verified.  Dividing by that size
        understates the rate of a three-instance run by a factor of 167.
        """
        write_report(tmp_path, "inst-0", resolved=True,
                     patch_successfully_applied=True)
        write_report(tmp_path, "inst-1", resolved=False,
                     patch_successfully_applied=True)
        write_report(tmp_path, "inst-2", resolved=False,
                     patch_successfully_applied=True)
        write_summary(
            tmp_path,
            total_instances=500,
            submitted_instances=3,
            resolved_instances=1,
            unresolved_instances=2,
        )

        metrics = collect_metrics(tmp_path)

        assert metrics["status"] == METRICS_STATUS_OK
        assert metrics["total_instances"] == 3
        assert metrics["dataset_instances"] == 500
        assert metrics["submitted_instances"] == 3
        assert metrics["resolved"] == 1
        assert metrics["unresolved"] == 2
        assert metrics["resolve_rate"] == 1 / 3

    def test_the_counts_add_up(self, tmp_path: Path) -> None:
        """Every count describes the same set of instances."""
        write_report(tmp_path, "inst-0", resolved=True,
                     patch_successfully_applied=True)
        write_report(tmp_path, "inst-1", resolved=False,
                     patch_successfully_applied=True)

        metrics = collect_metrics(tmp_path)

        assert metrics["resolved"] + metrics["unresolved"] == (
            metrics["total_instances"]
        )

    def test_a_summary_that_disagrees_is_reported_as_inconsistent(
        self, tmp_path: Path
    ) -> None:
        """Two sources that count the same thing cannot be silently merged."""
        write_report(tmp_path, "inst-0", resolved=True,
                     patch_successfully_applied=True)
        write_report(tmp_path, "inst-1", resolved=False,
                     patch_successfully_applied=True)
        write_summary(
            tmp_path,
            total_instances=500,
            submitted_instances=2,
            resolved_instances=2,
        )

        metrics = collect_metrics(tmp_path)

        assert metrics["status"] == METRICS_STATUS_INCONSISTENT
        assert metrics["notes"]

    def test_a_run_without_reports_is_not_counted_as_unresolved(
        self, tmp_path: Path
    ) -> None:
        """Predictions without a report were never evaluated."""
        write_summary(
            tmp_path,
            total_instances=500,
            submitted_instances=3,
            resolved_instances=0,
        )

        metrics = collect_metrics(tmp_path)

        assert metrics["status"] == METRICS_STATUS_NO_REPORTS
        assert metrics["total_instances"] == 0
        assert metrics["resolve_rate"] is None
        assert metrics["submitted_instances"] == 3

    def test_an_unreadable_report_counts_as_unresolved(
        self, tmp_path: Path
    ) -> None:
        """A report that cannot be read is counted, not dropped."""
        broken = tmp_path / "inst-0"
        broken.mkdir()
        (broken / "report.json").write_text("not json at all")

        metrics = collect_metrics(tmp_path)

        assert metrics["total_instances"] == 1
        assert metrics["resolved"] == 0
        assert metrics["unresolved"] == 1
        assert metrics["malformed_reports"] == 1

    def test_the_patch_count_comes_from_the_reports(self, tmp_path: Path) -> None:
        """The number of applied patches is counted from the reports."""
        write_report(tmp_path, "inst-0", resolved=True,
                     patch_successfully_applied=True)
        write_report(tmp_path, "inst-1", resolved=False,
                     patch_successfully_applied=False)

        metrics = collect_metrics(tmp_path)

        assert metrics["patch_applied"] == 1


class TestWriteManifest:
    """The manifest must let a teammate reconstruct the run."""

    def test_records_relative_paths(self, tmp_path: Path) -> None:
        """Artefact paths are relative to the run directory."""
        run_dir = tmp_path / "run-1"
        (run_dir / "run-agent").mkdir(parents=True)
        (run_dir / "run-agent" / "preds.json").write_text("{}")
        (run_dir / "config.json").write_text("{}")
        (run_dir / "metrics.json").write_text("{}")

        manifest_path = write_manifest(
            run_dir,
            {"run_id": "run-1", "created_at": "now"},
            {"status": METRICS_STATUS_OK, "total_instances": 1},
        )
        manifest = json.loads(manifest_path.read_text())

        assert manifest["artefacts"]["predictions"] == "run-agent/preds.json"
        assert manifest["artefacts"]["metrics"] == "metrics.json"
        assert manifest["config"] == "config.json"

    def test_records_absent_artefacts_as_null(self, tmp_path: Path) -> None:
        """An artefact that was not produced is recorded as null."""
        run_dir = tmp_path / "run-1"
        run_dir.mkdir()

        manifest_path = write_manifest(
            run_dir,
            {"run_id": "run-1", "created_at": "now"},
            {"status": METRICS_STATUS_NO_REPORTS},
        )
        manifest = json.loads(manifest_path.read_text())

        assert manifest["artefacts"]["predictions"] is None
        assert manifest["artefacts"]["eval_logs_dir"] is None

    def test_embeds_the_metrics_verbatim(self, tmp_path: Path) -> None:
        """The manifest and metrics.json cannot hold different numbers."""
        run_dir = tmp_path / "run-1"
        run_dir.mkdir()
        metrics = {"status": METRICS_STATUS_OK, "total_instances": 3,
                   "resolved": 1, "resolve_rate": 1 / 3}

        manifest_path = write_manifest(
            run_dir, {"run_id": "run-1", "created_at": "now"}, metrics
        )
        manifest = json.loads(manifest_path.read_text())

        assert manifest["metrics_summary"] == metrics
