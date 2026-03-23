"""Smoke tests for the microagent CLI."""
from typer.testing import CliRunner

from microagent.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "microagent" in result.output.lower()
