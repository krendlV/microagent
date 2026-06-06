# MCP Integration

MicroAgent exposes its full pipeline as a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server. This lets AI assistants like Claude drive segmentation workflows using natural language.

---

## What Is MCP?

MCP is an open protocol that lets AI assistants call external tools (functions) in a structured, safe way. When MicroAgent runs as an MCP server, Claude can:
- Inspect your image data
- Run segmentation with appropriate models
- Evaluate results and compare runs
- Generate reports
- Create and manage project configurations

No Python code is required—just describe what you want in plain language.

---

## Setup with Claude Code

### 1. Install MicroAgent with MCP support

```bash
pip install "microagent[mcp]"
```

### 2. Add to Claude Code settings

Edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "microagent": {
      "command": "uvx",
      "args": ["microagent[mcp]", "mcp-server"]
    }
  }
}
```

Or if installed locally:

```json
{
  "mcpServers": {
    "microagent": {
      "command": "microagent",
      "args": ["mcp-server"]
    }
  }
}
```

### 3. Restart Claude Code

The `microagent` tools will appear in Claude's tool list.

### 4. Start using it

```
Inspect the images in /data/experiment1 and tell me if there are any QC issues.
```

```
Segment the cells in /data/hela_images using the best model for fluorescence nuclei.
```

```
Compare the segmentation quality between my pretrained and fine-tuned models.
```

---

## Setup with Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "microagent": {
      "command": "microagent",
      "args": ["mcp-server"],
      "env": {}
    }
  }
}
```

Or in Cursor global settings (`~/.cursor/mcp.json`) to use across all projects.

---

## Setup with Other MCP Clients

MicroAgent follows the standard MCP spec. Start the server:

```bash
microagent mcp-server
```

The server communicates over stdin/stdout (stdio transport). Point any MCP-compatible client to this command.

For HTTP transport (experimental):

```bash
microagent mcp-server --transport streamable-http --port 8080
```

---

## Using MicroAgent with Any MCP Client or Local Model

MicroAgent is the MCP *server* — the LLM client is entirely your choice. The server makes no LLM calls of its own; all AI reasoning happens in the client. Any MCP-compatible client works, including fully open-source and locally-run stacks.

### Local stack: LibreChat + Ollama (fully air-gapped)

Run a privacy-preserving setup with no data leaving your machine:

1. Install [Ollama](https://ollama.ai/) and pull a model:
   ```bash
   ollama pull llama3
   ```

2. Install [LibreChat](https://www.librechat.ai/) and configure it to use the Ollama endpoint.

3. Add MicroAgent as an MCP server in `librechat.yaml`:
   ```yaml
   mcp:
     servers:
       microagent:
         command: microagent
         args: [mcp-server]
   ```

4. Start a conversation in LibreChat backed by your local Ollama model — it can call all MicroAgent tools exactly like Claude does.

### Local stack: Continue.dev + Ollama

[Continue](https://continue.dev/) is a VS Code/JetBrains extension with MCP support:

1. In `.continue/config.json` add:
   ```json
   {
     "mcpServers": [
       {
         "name": "microagent",
         "command": "microagent",
         "args": ["mcp-server"]
       }
     ]
   }
   ```

2. Point Continue to your Ollama model — no API keys required.

### EU/GDPR stack: Mistral via MCP-capable client

Mistral AI's API is hosted in Europe. Use it with any MCP-compatible client that supports Mistral models (LibreChat, Zed, OpenWebUI, etc.):

1. Get a [Mistral API key](https://console.mistral.ai/).

2. Configure your client to use `https://api.mistral.ai/v1` with your Mistral key.

3. Add MicroAgent as an MCP server — the server config is identical regardless of which LLM you use on the client side:
   ```json
   {
     "mcpServers": {
       "microagent": {
         "command": "microagent",
         "args": ["mcp-server"]
       }
     }
   }
   ```

> **Key point:** The MCP server config is the same for every LLM provider. Swap models freely without touching MicroAgent's configuration.

---

## Available Tools

### `inspect_data`

Quality-control check on an image directory.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image_dir` | string | yes | Path to image directory |

**Returns:** `{status, file_count, dimensions, channel_stats, qc_warnings}`

**Example:**
```json
{"image_dir": "/data/hela_images"}
```

---

### `segment`

Run cell segmentation.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image_dir` | string | yes | Path to image directory |
| `output_dir` | string | no | Output directory (default: `masks`) |
| `model` | string | no | `auto`, `cellpose`, `stardist`, `micro_sam` (default: `auto`) |
| `diameter` | number | no | Cell diameter in pixels (CellPose; null = auto) |

**Returns:** `{status, mask_paths, model_info, per_image_stats, elapsed_seconds}`

---

### `evaluate`

Compute segmentation metrics against ground truth.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `pred_dir` | string | yes | Predicted masks directory |
| `gt_dir` | string | yes | Ground truth masks directory |
| `thresholds` | string | no | Comma-separated IoU thresholds (default: `"0.5,0.75,0.9"`) |

**Returns:** `{status, summary: {f1, precision, recall, mean_f1, panoptic_quality}, per_image}`

---

### `train`

Fine-tune CellPose on labeled data.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image_dir` | string | yes | Training images directory |
| `gt_dir` | string | yes | Ground truth masks directory |
| `epochs` | integer | no | Training epochs (default: 100) |
| `output_dir` | string | no | Model save directory (default: `models`) |

**Returns:** `{status, model_path, train_losses, test_losses, best_epoch, elapsed_seconds}`

---

### `optimize`

Hyperparameter optimization with Optuna.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image_dir` | string | yes | Images directory |
| `gt_dir` | string | yes | Ground truth directory |
| `n_trials` | integer | no | Number of trials (default: 20) |
| `metric` | string | no | `f1`, `mean_f1`, `pq`, `precision`, `recall` (default: `f1`) |

**Returns:** `{status, best_params, best_value, baseline_value, improvement}`

---

### `generate_report`

Generate an HTML report from previous run results.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `output` | string | no | Output HTML path (default: `report.html`) |

**Returns:** `{status, report_path}`

---

### `get_project_info`

Read and return the current `project.yaml`.

**Parameters:** none

**Returns:** `{status, project: {...}}`

---

### `create_project`

Create a new `project.yaml`.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `organism` | string | yes | e.g. `human`, `mouse` |
| `modality` | string | yes | `fluorescence`, `brightfield`, `confocal`, `H&E`, etc. |
| `structures` | string | yes | Comma-separated, e.g. `nuclei,whole_cells` |
| `channels` | string | no | Channel mapping (default: `"0,0"`) |
| `image_format` | string | no | `tiff`, `ome-tiff`, `czi`, etc. (default: `tiff`) |

**Returns:** `{status, project: {...}, recommended_model}`

---

## Example Conversation Workflows

### Workflow 1: First-time analysis

```
User: I have fluorescence images of HeLa cells in /data/hela. Can you analyze them?

Claude: I'll start by inspecting the data for quality issues.
[calls inspect_data("/data/hela")]

Claude: Found 45 TIFF files, all 1024×1024, 16-bit. No QC warnings.
I'll create a project configuration and run segmentation.
[calls create_project(organism="human", modality="fluorescence", structures="nuclei,whole_cells")]
[calls segment("/data/hela", model="auto")]

Claude: Segmentation complete. Mean cell count: 52 per image.
Generating report...
[calls generate_report()]
Report saved to report.html.
```

### Workflow 2: Evaluating and improving

```
User: My segmentation doesn't look right for my phase contrast images.

Claude: Let me check your current results and optimize the parameters.
[calls evaluate("masks/", "ground_truth/")]

Claude: Current F1 = 0.61. Let me try hyperparameter optimization.
[calls optimize("images/", "ground_truth/", n_trials=30, metric="f1")]

Claude: Optimization found better parameters (diameter=22, flow_threshold=0.3).
F1 improved from 0.61 → 0.79. Update your project.yaml with these values.
```

### Workflow 3: Fine-tuning

```
User: I have 30 manually annotated images. Can you fine-tune the model?

Claude: I'll train a custom model on your annotations.
[calls train("annotated_images/", "annotations/", epochs=200)]

Claude: Training complete. Test F1 = 0.88 (vs. 0.71 pretrained).
Model saved to models/cellpose_finetuned.pth.
```
