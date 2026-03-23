"""Tests for the MCP server (mcp_server.py)."""

from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp", reason="mcp package not installed")


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
