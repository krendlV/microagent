---
name: microagent
description: Open-source microscopy image analysis with MCP server integration. Provides segmentation, evaluation, training, and reporting for microscopy images using CellPose, StarDist, and micro-SAM models.
user-invocable: true
allowed-tools: bash read_file write_file grep search_replace
---

# MicroAgent Vibe Skill

You are an expert in microscopy image analysis using MicroAgent. This skill provides access to automated microscopy image segmentation, evaluation, training, and reporting through the Model Context Protocol (MCP).

## When to Use This Skill

Use this skill when the user wants to:
- Analyze microscopy images (segmentation, QC, evaluation)
- Work with CellPose, StarDist, or micro-SAM models
- Generate HTML reports from microscopy data
- Fine-tune segmentation models on custom data
- Optimize hyperparameters for segmentation
- Run end-to-end microscopy analysis pipelines

## MCP Server Setup

MicroAgent provides an MCP server that exposes all its functionality to Vibe. To use it:

1. **Install microagent with MCP support:**
   ```bash
   pip install 'microagent[mcp]'
   ```

2. **Add to Vibe config (~/.vibe/config.toml):**
   ```toml
   [[mcp_servers]]
   name = "microagent"
   command = "microagent"
   args = ["mcp-server"]
   ```

3. **Reload Vibe:** Use `/reload` command

## Available MCP Tools

Once connected, the following tools are available through MCP:

- `inspect_data(path)` - Inspect directory of microscopy images, returns QC report
- `segment(image_dir, output_dir, model, diameter)` - Run segmentation
- `evaluate(pred_dir, gt_dir, thresholds)` - Evaluate segmentation quality
- `train(image_dir, gt_dir, epochs, output_dir)` - Fine-tune CellPose model
- `optimize(image_dir, gt_dir, n_trials, metric)` - Run hyperparameter optimization
- `generate_report(output)` - Generate HTML report
- `get_project_info()` - Read current project.yaml
- `create_project(organism, modality, structures, channels, image_format)` - Create project.yaml

## CLI Commands (when MCP not available)

If MCP is not configured, you can use the CLI directly:

- `microagent inspect /path/to/images` - Run QC inspection
- `microagent segment /path/to/images` - Segment images
- `microagent evaluate /path/to/preds /path/to/gt` - Evaluate masks
- `microagent train /images /gt_masks` - Fine-tune model
- `microagent optimize /images /gt_masks` - Optimize hyperparameters
- `microagent report` - Generate HTML report
- `microagent init` - Interactive project setup
- `microagent demo` - Run full pipeline with synthetic data
- `microagent export` - Export reproducibility bundle

## Project Configuration

MicroAgent uses `project.yaml` for configuration:

```yaml
organism: "Homo sapiens"
modality: "fluorescence"
structures: ["nucleus", "cytoplasm"]
imaging:
  format: "tiff"
  channels:
    cytoplasm: 0
    nucleus: 1
```

## Model Selection

Auto-selection based on project.yaml, or specify manually:
- `auto` - Automatic based on project metadata
- `cellpose` - CellPose models (cyto2, cyto3, cpsam)
- `stardist` - StarDist models (2D_versatile_fluo, 2D_versatile_he)
- `sam` - micro-SAM (Segment Anything)

## Best Practices

1. Always run `inspect` before segmentation to check image quality
2. Use `project.yaml` for reproducibility
3. Start with `demo` command to understand the full pipeline
4. For GPU acceleration, ensure CUDA and cuDNN are installed
5. Large images may need to be tiled or downscaled

## Common Workflows

### Quick Start
1. `microagent init --data-dir /path/to/images` - Create project.yaml
2. `microagent inspect /path/to/images` - QC check
3. `microagent segment /path/to/images` - Generate masks
4. `microagent report` - Generate HTML report

### Full Evaluation Pipeline
1. Create ground truth masks
2. `microagent segment /images` - Run segmentation
3. `microagent evaluate /masks /ground_truth` - Compare against GT
4. `microagent optimize /images /ground_truth` - Tune parameters
5. `microagent train /images /ground_truth` - Fine-tune model

### Batch Processing
Use the MCP server for programmatic access to all functions.
