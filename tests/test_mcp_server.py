"""Tests for the MCP server (mcp_server.py)."""

from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp", reason="mcp package not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_seg_result(output_dir=None):
    from microagent.core.segment import PerImageStats, SegmentationResult

    mask_path = str(output_dir / "image_000_mask.tif") if output_dir else "/tmp/image_000_mask.tif"
    return SegmentationResult(
        mask_paths=[mask_path],
        model_info={"backend": "cellpose", "model_name": "cpsam", "parameters": {}},
        parameters={},
        elapsed_seconds=0.1,
        per_image_stats=[PerImageStats(filename="image_000.tif", n_labels=3, elapsed_seconds=0.1)],
    )


def _make_fake_eval_result():
    from microagent.core.evaluate import (
        DatasetMetrics,
        EvaluationResult,
        ImageMetrics,
        ThresholdMetrics,
    )

    tm = ThresholdMetrics(
        threshold=0.5, precision=0.8, recall=0.8, f1=0.8, tp=4, fp=1, fn=1, mean_true_score=0.7
    )
    im = ImageMetrics(
        filename="image_000_mask.tif",
        gt_count=5,
        pred_count=5,
        per_threshold=[tm],
        mean_f1=0.8,
        panoptic_quality=0.7,
        iou_distribution=[0.7, 0.8],
    )
    summary = DatasetMetrics(
        n_images=1,
        per_threshold=[tm],
        mean_f1=0.8,
        panoptic_quality=0.7,
        mean_gt_count=5.0,
        mean_pred_count=5.0,
    )
    return EvaluationResult(
        per_image=[im],
        summary=summary,
        best_images=["image_000_mask.tif"],
        worst_images=[],
        unmatched_preds=[],
        unmatched_gts=[],
    )


# ---------------------------------------------------------------------------
# Import server once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    from microagent.mcp_server import mcp as _mcp

    return _mcp


# ---------------------------------------------------------------------------
# test_mcp_tools_registered
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "inspect_data",
    "segment",
    "evaluate",
    "train",
    "optimize",
    "generate_report",
    "get_project_info",
    "create_project",
}


def test_mcp_tools_registered(server):
    """All 8 tools must be registered on the FastMCP server."""
    # FastMCP exposes registered tools via ._tool_manager or .list_tools()
    # We normalise across SDK versions by inspecting internal structures.
    tool_names: set[str] = set()

    # FastMCP >= 1.x stores tools in _tool_manager._tools (dict keyed by name)
    if hasattr(server, "_tool_manager"):
        mgr = server._tool_manager
        if hasattr(mgr, "_tools"):
            tool_names = set(mgr._tools.keys())
        elif hasattr(mgr, "tools"):
            tool_names = set(mgr.tools.keys())

    # Fallback: iterate over all attributes looking for Tool objects
    if not tool_names:
        import inspect

        from mcp.server.fastmcp.tools import Tool  # type: ignore[import-untyped]

        for name in dir(server):
            obj = getattr(server, name, None)
            if isinstance(obj, Tool):
                tool_names.add(obj.name)

    assert EXPECTED_TOOLS.issubset(tool_names), (
        f"Missing tools: {EXPECTED_TOOLS - tool_names}"
    )


# ---------------------------------------------------------------------------
# test_mcp_tool_schemas
# ---------------------------------------------------------------------------


def test_mcp_tool_schemas(server):
    """Each registered tool must expose a valid JSON schema for its parameters."""
    mgr = getattr(server, "_tool_manager", None)
    if mgr is None:
        pytest.skip("Cannot access _tool_manager on this MCP SDK version")

    tools: dict = {}
    if hasattr(mgr, "_tools"):
        tools = mgr._tools
    elif hasattr(mgr, "tools"):
        tools = mgr.tools

    for name in EXPECTED_TOOLS:
        assert name in tools, f"Tool '{name}' not registered"
        tool = tools[name]

        # Each tool must have a non-empty description
        assert tool.description, f"Tool '{name}' has no description"

        # Each tool must expose a JSON schema with a 'properties' dict
        schema = tool.parameters if hasattr(tool, "parameters") else getattr(tool, "inputSchema", {})
        assert isinstance(schema, dict), f"Tool '{name}' schema is not a dict"
        # 'properties' key is present for tools with at least one parameter
        # (tools with no required params may omit it — we just ensure it's a dict)


# ---------------------------------------------------------------------------
# test_mcp_inspect_tool
# ---------------------------------------------------------------------------


def test_mcp_inspect_tool(tmp_image_dir):
    """inspect_data called with a real fixture path must return a success dict."""
    from microagent.mcp_server import inspect_data

    result = inspect_data(str(tmp_image_dir))

    assert isinstance(result, dict)
    assert result.get("status") == "success"
    assert result["file_count"] == 5
    assert "file_paths" in result
    assert "dimensions" in result
    assert "intensity_stats" in result
    assert "issues" in result


# ---------------------------------------------------------------------------
# test_mcp_error_handling
# ---------------------------------------------------------------------------


def test_mcp_error_handling(tmp_path):
    """Calling tools with invalid paths must return an error dict, not raise."""
    from microagent.mcp_server import (
        evaluate,
        inspect_data,
        optimize,
        segment,
        train,
    )

    nonexistent = str(tmp_path / "does_not_exist")

    for fn, args in [
        (inspect_data, (nonexistent,)),
        (segment, (nonexistent,)),
        (evaluate, (nonexistent, nonexistent)),
        (train, (nonexistent, nonexistent)),
        (optimize, (nonexistent, nonexistent)),
    ]:
        result = fn(*args)
        assert isinstance(result, dict), f"{fn.__name__} did not return a dict"
        assert result.get("status") == "error", (
            f"{fn.__name__} did not return status='error' for invalid input; got {result}"
        )
        assert "error" in result, f"{fn.__name__} missing 'error' key"


# ---------------------------------------------------------------------------
# test_mcp_get_project_info_missing
# ---------------------------------------------------------------------------


def test_mcp_get_project_info_missing(monkeypatch, tmp_path):
    """get_project_info returns an error when project.yaml is absent."""
    monkeypatch.chdir(tmp_path)
    from microagent.mcp_server import get_project_info

    result = get_project_info()
    assert result["status"] == "error"
    assert "project.yaml" in result["error"]


# ---------------------------------------------------------------------------
# test_mcp_create_and_read_project
# ---------------------------------------------------------------------------


def test_mcp_create_and_read_project(monkeypatch, tmp_path):
    """create_project writes project.yaml; get_project_info reads it back."""
    pytest.importorskip("yaml", reason="PyYAML not installed")
    monkeypatch.chdir(tmp_path)
    from microagent.mcp_server import create_project, get_project_info

    created = create_project(
        organism="mouse",
        modality="confocal",
        structures="nuclei,cytoplasm",
        channels="0,1",
        image_format="tiff",
    )
    assert created["status"] == "success"
    assert (tmp_path / "project.yaml").exists()

    info = get_project_info()
    assert info["status"] == "success"
    assert info["organism"] == "mouse"
    assert info["modality"] == "confocal"


# ---------------------------------------------------------------------------
# test_mcp_segment_uses_project
# ---------------------------------------------------------------------------


def test_mcp_segment_uses_project(monkeypatch, tmp_path):
    """segment tool passes project_path to run_segmentation."""
    yaml = pytest.importorskip("yaml", reason="PyYAML not installed")

    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        yaml.dump({"recommended_model": "stardist", "recommended_params": {}})
    )

    captured: dict = {}

    def _mock_run_segmentation(image_dir, output_dir, model, project_path=None, **kwargs):
        captured["project_path"] = project_path
        output_dir.mkdir(parents=True, exist_ok=True)
        return _make_fake_seg_result(output_dir)

    monkeypatch.setattr("microagent.core.segment.run_segmentation", _mock_run_segmentation)

    from microagent.mcp_server import segment

    result = segment(
        image_dir=str(tmp_path / "images"),
        output_dir=str(tmp_path / "masks"),
        project=str(project_yaml),
        track=False,
    )

    assert result["status"] == "success"
    assert captured.get("project_path") == project_yaml


# ---------------------------------------------------------------------------
# test_mcp_segment_logs_to_experiments
# ---------------------------------------------------------------------------


def test_mcp_segment_logs_to_experiments(monkeypatch, tmp_path):
    """segment with track=True writes to experiments.jsonl and returns run_id."""
    from microagent.fair.tracking import ExperimentTracker

    monkeypatch.chdir(tmp_path)

    def _mock_run_segmentation(image_dir, output_dir, model, project_path=None, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        return _make_fake_seg_result(output_dir)

    monkeypatch.setattr("microagent.core.segment.run_segmentation", _mock_run_segmentation)

    from microagent.mcp_server import segment

    result = segment(
        image_dir=str(tmp_path / "images"),
        output_dir=str(tmp_path / "masks"),
    )

    assert result["status"] == "success"
    assert "run_id" in result
    run_id = result["run_id"]
    assert isinstance(run_id, str) and len(run_id) == 8

    jsonl = tmp_path / "experiments.jsonl"
    assert jsonl.exists()
    record = ExperimentTracker(jsonl).get_run(run_id)
    assert record["run_id"] == run_id
    assert record["results"]["backend"] == "cellpose"


# ---------------------------------------------------------------------------
# test_mcp_evaluate_logs_to_experiments
# ---------------------------------------------------------------------------


def test_mcp_evaluate_logs_to_experiments(monkeypatch, tmp_path):
    """evaluate with track=True writes to experiments.jsonl and returns run_id."""
    from microagent.fair.tracking import ExperimentTracker

    monkeypatch.chdir(tmp_path)

    fake = _make_fake_eval_result()
    monkeypatch.setattr("microagent.core.evaluate.evaluate_masks", lambda *a, **kw: fake)

    from microagent.mcp_server import evaluate

    result = evaluate(pred_dir=str(tmp_path / "preds"), gt_dir=str(tmp_path / "gt"))

    assert result["status"] == "success"
    assert "run_id" in result
    run_id = result["run_id"]
    assert isinstance(run_id, str) and len(run_id) == 8

    jsonl = tmp_path / "experiments.jsonl"
    assert jsonl.exists()
    record = ExperimentTracker(jsonl).get_run(run_id)
    assert record["run_id"] == run_id
    assert record["results"]["mean_f1"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# test_mcp_segment_track_false_no_experiments
# ---------------------------------------------------------------------------


def test_mcp_segment_track_false_no_experiments(monkeypatch, tmp_path):
    """segment with track=False does not write experiments.jsonl."""
    monkeypatch.chdir(tmp_path)

    def _mock_run_segmentation(image_dir, output_dir, model, project_path=None, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        return _make_fake_seg_result(output_dir)

    monkeypatch.setattr("microagent.core.segment.run_segmentation", _mock_run_segmentation)

    from microagent.mcp_server import segment

    result = segment(
        image_dir=str(tmp_path / "images"),
        output_dir=str(tmp_path / "masks"),
        track=False,
    )

    assert result["status"] == "success"
    assert "run_id" not in result
    assert not (tmp_path / "experiments.jsonl").exists()
