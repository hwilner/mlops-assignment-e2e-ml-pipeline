"""Tests for the command construction in :mod:`pipeline.evaluation`."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import build_run_config
from pipeline.evaluation import (
    agent_config_path,
    build_agent_command,
    build_eval_command,
    resolve_dataset_name,
)


@pytest.fixture
def fake_package_root(tmp_path: Path) -> Path:
    """Create a directory that looks like an installed mini-swe-agent.

    Args:
        tmp_path: The temporary directory the fixture is built in.

    Returns:
        The root of the fake package.
    """
    config = tmp_path / "minisweagent" / "config" / "benchmarks" / "swebench.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("agent:\n  cost_limit: 3.0\n")
    return tmp_path / "minisweagent"


class TestAgentCommand:
    """The agent command has to match the mini-swe-agent interface."""

    def test_does_not_pass_a_cost_limit_option(
        self, fake_package_root: Path
    ) -> None:
        """The batch subcommand has no --cost-limit option.

        Passing it makes mini-swe-agent exit with ``No such option:
        --cost-limit`` before a single instance is attempted.
        """
        command = build_agent_command(
            {
                "subset": "verified",
                "split": "test",
                "model": "nebius/moonshotai/Kimi-K2.6",
                "task_slice": "0:3",
                "workers": 3,
                "cost_limit": 1.5,
            },
            output_dir="/tmp/agent",
            package_root=fake_package_root,
        )

        assert "--cost-limit" not in command

    def test_passes_the_cost_limit_as_a_configuration_override(
        self, fake_package_root: Path
    ) -> None:
        """The limit is an agent configuration key."""
        command = build_agent_command(
            {
                "subset": "verified",
                "split": "test",
                "model": "nebius/moonshotai/Kimi-K2.6",
                "task_slice": "0:3",
                "workers": 3,
                "cost_limit": 1.5,
            },
            output_dir="/tmp/agent",
            package_root=fake_package_root,
        )

        assert "agent.cost_limit=1.5" in command

    def test_always_passes_the_cost_limit(
        self, fake_package_root: Path
    ) -> None:
        """A zero limit is passed as well.

        The benchmark configuration mini-swe-agent ships with caps each task
        at 3 US dollars, so a zero limit that is not passed would silently
        apply a cap that nobody asked for.
        """
        command = build_agent_command(
            {
                "subset": "verified",
                "split": "test",
                "model": "nebius/moonshotai/Kimi-K2.6",
                "task_slice": "0:3",
                "workers": 3,
                "cost_limit": 0,
            },
            output_dir="/tmp/agent",
            package_root=fake_package_root,
        )

        assert "agent.cost_limit=0" in command

    def test_names_the_benchmark_configuration(
        self, fake_package_root: Path
    ) -> None:
        """The default configuration is named again.

        mini-swe-agent drops its default configuration as soon as any
        --config value is given, so an override without the benchmark
        configuration would run the agent without its instructions.
        """
        command = build_agent_command(
            {
                "subset": "verified",
                "split": "test",
                "model": "nebius/moonshotai/Kimi-K2.6",
                "task_slice": "0:3",
                "workers": 3,
                "cost_limit": 0,
            },
            output_dir="/tmp/agent",
            package_root=fake_package_root,
        )

        assert str(fake_package_root / "config" / "benchmarks" / "swebench.yaml") in command

    def test_carries_the_run_parameters(self, fake_package_root: Path) -> None:
        """Every parameter of the run reaches the agent."""
        command = build_agent_command(
            {
                "subset": "verified",
                "split": "test",
                "model": "nebius/moonshotai/Kimi-K2.6",
                "task_slice": "0:5",
                "workers": 3,
                "cost_limit": 0,
            },
            output_dir="/tmp/agent",
            package_root=fake_package_root,
        )

        assert command[:4] == ["uv", "run", "mini-extra", "swebench"]
        assert command[command.index("--subset") + 1] == "verified"
        assert command[command.index("--split") + 1] == "test"
        assert command[command.index("--slice") + 1] == "0:5"
        assert command[command.index("--workers") + 1] == "3"
        assert command[command.index("-o") + 1] == "/tmp/agent"

    def test_reports_a_missing_configuration(self, tmp_path: Path) -> None:
        """A missing benchmark configuration is reported, not guessed."""
        with pytest.raises(FileNotFoundError, match="does not ship"):
            agent_config_path(tmp_path / "empty")


class TestEvalCommand:
    """The evaluation command has to match the SWE-bench harness interface."""

    def test_passes_the_split(self) -> None:
        """The harness must evaluate the split the agent ran.

        The harness falls back to its own default split, and a prediction
        for an instance of another split matches nothing, so the harness
        would report on no instance at all.
        """
        command = build_eval_command(
            {
                "run_id": "run-1",
                "split": "dev",
                "subset": "verified",
                "workers": 3,
            },
            preds_path="preds.json",
            report_dir="run-eval",
        )

        assert command[command.index("--split") + 1] == "dev"

    def test_resolves_the_dataset_from_the_subset(self) -> None:
        """The subset selects the dataset the harness loads."""
        command = build_eval_command(
            {
                "run_id": "run-1",
                "split": "test",
                "subset": "lite",
                "workers": 3,
            },
            preds_path="preds.json",
            report_dir="run-eval",
        )

        assert command[command.index("--dataset_name") + 1] == (
            "princeton-nlp/SWE-bench_Lite"
        )

    def test_carries_the_run_parameters(self) -> None:
        """Every parameter of the run reaches the harness."""
        command = build_eval_command(
            {
                "run_id": "run-1",
                "split": "test",
                "subset": "verified",
                "workers": 3,
            },
            preds_path="/tmp/preds.json",
            report_dir="/tmp/run-eval",
        )

        assert command[command.index("--predictions_path") + 1] == "/tmp/preds.json"
        assert command[command.index("--max_workers") + 1] == "3"
        assert command[command.index("--run_id") + 1] == "run-1"
        assert command[command.index("--report_dir") + 1] == "/tmp/run-eval"


def test_the_agent_command_carries_the_run_parameters(tmp_path: Path) -> None:
    """The agent command is built from the run configuration."""
    run_config = build_run_config(
        {
            "split": "dev",
            "subset": "lite",
            "workers": 2,
            "cost_limit": 0.5,
            "run_id": "run-1",
        }
    )
    package_root = tmp_path / "minisweagent"
    config = package_root / "config" / "benchmarks" / "swebench.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("agent:\n  cost_limit: 3.0\n")

    command = build_agent_command(
        run_config, output_dir=tmp_path / "out", package_root=package_root
    )

    assert command[command.index("--split") + 1] == "dev"
    assert command[command.index("--subset") + 1] == "lite"
    assert command[command.index("--workers") + 1] == "2"
    assert "agent.cost_limit=0.5" in command


class TestResolveDatasetName:
    """The subset names the dataset the harness loads."""

    def test_maps_the_known_subsets(self) -> None:
        """A known subset maps to its dataset."""
        assert resolve_dataset_name("verified") == (
            "princeton-nlp/SWE-bench_Verified"
        )
        assert resolve_dataset_name("lite") == "princeton-nlp/SWE-bench_Lite"
        assert resolve_dataset_name("full") == "princeton-nlp/SWE-bench"

    def test_is_case_insensitive(self) -> None:
        """The subset name is matched case-insensitively."""
        assert resolve_dataset_name("Verified") == (
            "princeton-nlp/SWE-bench_Verified"
        )

    def test_passes_an_unknown_subset_through(self) -> None:
        """A fully-qualified dataset name can be supplied directly."""
        assert resolve_dataset_name("my-org/my-dataset") == "my-org/my-dataset"
