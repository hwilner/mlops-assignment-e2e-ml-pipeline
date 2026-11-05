"""Tests for the prediction checks in :mod:`pipeline.predictions`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.predictions import read_predictions


class TestReadPredictions:
    """A broken prediction file has to be caught before the harness runs."""

    def test_reads_the_predictions(self, tmp_path: Path) -> None:
        """A valid prediction file is returned keyed by instance."""
        preds_path = tmp_path / "preds.json"
        preds_path.write_text(json.dumps({"inst-0": {"model_patch": "diff"}}))

        predictions = read_predictions(preds_path)

        assert list(predictions) == ["inst-0"]

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        """A missing prediction file names the path it looked for."""
        with pytest.raises(FileNotFoundError, match=r"preds\.json not found"):
            read_predictions(tmp_path / "preds.json")

    def test_rejects_a_malformed_file(self, tmp_path: Path) -> None:
        """A file that is not JSON is reported as such."""
        preds_path = tmp_path / "preds.json"
        preds_path.write_text("not json at all")

        with pytest.raises(ValueError, match="not valid JSON"):
            read_predictions(preds_path)

    def test_rejects_a_file_that_is_not_an_object(
        self, tmp_path: Path
    ) -> None:
        """The harness expects an object keyed by instance."""
        preds_path = tmp_path / "preds.json"
        preds_path.write_text(json.dumps(["inst-0"]))

        with pytest.raises(ValueError, match="expected a JSON object"):
            read_predictions(preds_path)

    def test_rejects_an_empty_prediction_set(self, tmp_path: Path) -> None:
        """A run that produced no patch has nothing to evaluate."""
        preds_path = tmp_path / "preds.json"
        preds_path.write_text(json.dumps({}))

        with pytest.raises(ValueError, match="holds no prediction"):
            read_predictions(preds_path)
