"""Airflow DAG: mini-swe-bench-single — run the agent on one instance.

This is the starter DAG that ships with the assignment.  It is kept because
the README asks every contributor to try it while setting the project up.

The subprocess call sets ``check=True``: without it a failing agent run is
reported to Airflow as a successful task, which hides exactly the failure
the setup instructions are meant to surface.

Example:
    Open http://localhost:8080, find "mini-swe-bench-single", click the ▶
    button, and click "Trigger".
"""

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dag(
    dag_id="mini-swe-bench-single",
    start_date=datetime(2024, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
)
def my_dag():
    @task
    def run_script():
        subprocess.run(
            [
                "uv",
                "run",
                "mini-extra",
                "swebench-single",
                "--subset",
                "verified",
                "--split",
                "test",
                "--model",
                "nebius/moonshotai/Kimi-K2.6",
                "--yolo",
                "--cost-limit",
                "0",
                "-i",
                "sympy__sympy-15599",
                "-o",
                "trajectory.json",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            env={
                **os.environ,
                "MSWEA_COST_TRACKING": "ignore_errors",
            },
        )

    run_script()


my_dag()
