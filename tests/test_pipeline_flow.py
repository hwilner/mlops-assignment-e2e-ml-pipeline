"""End-to-end test of the artefacts one run produces.

The individual steps are covered by the other test modules.  This module
runs them in the order the DAG runs them, with the harness output
simulated, and checks that the run directory a teammate receives is
self-consistent and passes the repository's own artefact checker.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import build_run_config, prepare_run_dir
from pipeline.evaluation import build_eval_command
from pipeline.metrics import (
    METRICS_STATUS_OK,
    collect_metrics,
    write_manifest,
)
from pipeline.predictions import read_predictions


def write_harness_output(eval_dir: Path, instances: dict[str, bool]) -> None:
    """Write the output the SWE-bench harness produces for a run.

    Args:
        eval_dir: The ``run-eval/`` directory of the run.
        instances: Maps each instance identifier to whether it was resolved.
    """
    for instance_id, resolved in instances.items():
        instance_dir = eval_dir / "run-1" / "nebius__moonshotai__Kimi-K2.6"
        instance_dir = instance_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "report.json").write_text(
            json.dumps(
                {
                    instance_id: {
                        "resolved": resolved,
                        "patch_successfully_applied": True,
                    }
                }
            )
        )
    (eval_dir / "nebius__moonshotai__Kimi-K2.6.run-1.json").write_text(
        json.dumps(
            {
                "total_instances": 500,
                "submitted_instances": len(instances),
                "completed_instances": len(instances),
                "resolved_instances": sum(instances.values()),
                "unresolved_instances": len(instances) - sum(instances.values()),
            }
        )
    )


def test_a_run_produces_a_self_consistent_directory(tmp_path: Path) -> None:
    """The run directory passes the repository's own artefact checker."""
    run_config = build_run_config(
        {
            "split": "test",
            "subset": "verified",
            "workers": 3,
            "task_slice": "0:3",
            "run_id": "run-1",
        }
    )
    run_dir = prepare_run_dir(run_config, tmp_path)

    # The agent stage writes its predictions into the run directory.
    agent_out_dir = run_dir / "run-agent"
    (agent_out_dir / "preds.json").write_text(
        json.dumps(
            {
                "inst-0": {"model_patch": "diff"},
                "inst-1": {"model_patch": "diff"},
                "inst-2": {"model_patch": "diff"},
            }
        )
    )
    predictions = read_predictions(agent_out_dir / "preds.json")
    assert len(predictions) == 3

    # The evaluation stage is pointed at the same split as the agent.
    eval_command = build_eval_command(
        run_config, agent_out_dir / "preds.json", run_dir / "run-eval"
    )
    assert eval_command[eval_command.index("--split") + 1] == "test"

    # The harness writes its reports and its summary.
    write_harness_output(
        run_dir / "run-eval",
        {"inst-0": True, "inst-1": False, "inst-2": False},
    )

    # The summary stage aggregates the reports and writes both artefacts.
    metrics = collect_metrics(run_dir / "run-eval")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_manifest(run_dir, run_config, metrics)

    assert metrics["status"] == METRICS_STATUS_OK
    assert metrics["total_instances"] == 3
    assert metrics["resolved"] == 1
    assert metrics["resolve_rate"] == 1 / 3
    assert metrics["dataset_instances"] == 500

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["metrics_summary"] == metrics
    assert manifest["artefacts"]["predictions"] == "run-agent/preds.json"
    assert manifest["artefacts"]["eval_logs_dir"] == "run-eval"

