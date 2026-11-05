"""Reading and checking the predictions produced by the agent.

The agent writes ``preds.json`` as an object keyed by instance identifier.
The evaluation harness accepts that file without complaint, so an empty or
malformed one has to be caught before the harness runs: otherwise the
harness reports on no instance and the run looks like a run that resolved
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_predictions(preds_path: Path) -> dict[str, Any]:
    """Read and sanity-check the predictions produced by the agent.

    Args:
        preds_path: Path of the ``preds.json`` file produced by the agent.

    Returns:
        The parsed predictions, keyed by instance identifier.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a JSON object, or holds no
            prediction at all.
    """
    preds_path = Path(preds_path)
    if not preds_path.exists():
        raise FileNotFoundError(
            f"preds.json not found at {preds_path}. "
            "The run_agent task may have failed silently."
        )
    try:
        predictions = json.loads(preds_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"preds.json at {preds_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(predictions, dict):
        raise ValueError(
            f"preds.json at {preds_path} holds a "
            f"{type(predictions).__name__}, expected a JSON object keyed by "
            "instance identifier."
        )
    if not predictions:
        raise ValueError(
            f"preds.json at {preds_path} holds no prediction. The agent "
            "produced no patch for any instance, so there is nothing to "
            "evaluate."
        )
    return predictions
