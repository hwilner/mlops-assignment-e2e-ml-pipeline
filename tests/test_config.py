"""Tests for the run-configuration handling in :mod:`pipeline.config`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.config import (
    DEFAULT_COST_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_TASK_SLICE,
    build_run_config,
    prepare_run_dir,
    validate_run_id,
)


class TestValidateRunId:
    """The run identifier becomes a directory name, so it must be safe."""

    @pytest.mark.parametrize(
        "run_id",
        ["run-2026-07-06-01", "run_1", "a", "Run.1", "0" * 128],
    )
    def test_accepts_safe_identifiers(self, run_id: str) -> None:
        """A safe identifier is returned unchanged."""
        assert validate_run_id(run_id) == run_id

    @pytest.mark.parametrize(
        "run_id",
        [
            "../../etc",
            "..",
            ".",
            "runs/../../escape",
            "/absolute/path",
            "sub/dir",
            "back\\slash",
            "",
            "   ",
            "-leading-dash",
            ".hidden",
            "has space",
        ],
    )
    def test_rejects_unsafe_identifiers(self, run_id: str) -> None:
        """An identifier that could escape the runs root is rejected."""
        with pytest.raises(ValueError):
            validate_run_id(run_id)

    def test_rejects_a_non_string(self) -> None:
        """A non-string identifier is rejected rather than coerced."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_run_id(42)

    def test_rejects_an_over_long_identifier(self) -> None:
        """An identifier longer than the file-name limit is rejected."""
        with pytest.raises(ValueError, match="at most 128 characters"):
            validate_run_id("a" * 129)


class TestBuildRunConfig:
    """The configuration is validated before any subprocess is started."""

    def test_applies_the_documented_defaults(self) -> None:
        """Optional parameters fall back to their documented defaults."""
        config = build_run_config(
            {"split": "test", "subset": "verified", "workers": 3}
        )
        assert config["model"] == DEFAULT_MODEL
        assert config["task_slice"] == DEFAULT_TASK_SLICE
        assert config["cost_limit"] == DEFAULT_COST_LIMIT
        assert config["split"] == "test"
        assert config["subset"] == "verified"
        assert config["workers"] == 3

    def test_requires_the_mandatory_parameters(self) -> None:
        """A missing mandatory parameter is reported by name."""
        with pytest.raises(ValueError, match="'split' is missing"):
            build_run_config({"subset": "verified", "workers": 3})

    def test_generates_a_run_id_when_none_is_given(self) -> None:
        """A generated identifier is a safe directory name."""
        config = build_run_config(
            {"split": "test", "subset": "verified", "workers": 3}
        )
        assert validate_run_id(config["run_id"]) == config["run_id"]

    def test_rejects_an_unsafe_run_id(self) -> None:
        """A supplied identifier is validated like a generated one."""
        with pytest.raises(ValueError, match="not a valid run directory name"):
            build_run_config(
                {
                    "split": "test",
                    "subset": "verified",
                    "workers": 3,
                    "run_id": "../../escape",
                }
            )

    @pytest.mark.parametrize("workers", [0, -1])
    def test_rejects_a_non_positive_worker_count(self, workers: int) -> None:
        """Zero workers would evaluate nothing and look like a completed run."""
        with pytest.raises(ValueError, match="at least 1"):
            build_run_config(
                {"split": "test", "subset": "verified", "workers": workers}
            )

    def test_accepts_a_worker_count_given_as_a_string(self) -> None:
        """A worker count supplied as a string is converted."""
        config = build_run_config(
            {"split": "test", "subset": "verified", "workers": "4"}
        )
        assert config["workers"] == 4

    def test_rejects_a_fractional_worker_count(self) -> None:
        """A fractional worker count is rejected rather than truncated."""
        with pytest.raises(ValueError, match="whole number"):
            build_run_config(
                {"split": "test", "subset": "verified", "workers": 2.5}
            )

    def test_rejects_a_boolean_worker_count(self) -> None:
        """A boolean is not a worker count, even though it is an int."""
        with pytest.raises(ValueError, match="boolean"):
            build_run_config(
                {"split": "test", "subset": "verified", "workers": True}
            )

    def test_rejects_a_negative_cost_limit(self) -> None:
        """A negative cost limit is rejected."""
        with pytest.raises(ValueError, match="must not be negative"):
            build_run_config(
                {
                    "split": "test",
                    "subset": "verified",
                    "workers": 3,
                    "cost_limit": -1,
                }
            )

    def test_records_the_creation_time(self) -> None:
        """The creation time is recorded in UTC with a trailing Z."""
        config = build_run_config(
            {"split": "test", "subset": "verified", "workers": 3}
        )
        assert config["created_at"].endswith("Z")
        assert "T" in config["created_at"]


class TestPrepareRunDir:
    """The run directory is created before any stage writes into it."""

    def test_creates_the_run_tree_and_writes_the_config(
        self, tmp_path: Path
    ) -> None:
        """The directory tree and config.json are created."""
        config = build_run_config(
            {"split": "test", "subset": "verified", "workers": 3}
        )
        run_dir = prepare_run_dir(config, tmp_path)

        assert run_dir == tmp_path / config["run_id"]
        assert (run_dir / "run-agent").is_dir()
        assert (run_dir / "run-eval").is_dir()

        written = json.loads((run_dir / "config.json").read_text())
        assert written == config

    def test_writes_the_run_inside_the_runs_root(
        self, tmp_path: Path
    ) -> None:
        """The run directory never escapes the runs root."""
        config = build_run_config(
            {
                "split": "test",
                "subset": "verified",
                "workers": 3,
                "run_id": "run-1",
            }
        )
        run_dir = prepare_run_dir(config, tmp_path)
        assert run_dir.parent == tmp_path
