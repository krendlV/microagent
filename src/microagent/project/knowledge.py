"""Project knowledge and YAML I/O for microscopy analysis projects."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from microagent.project.schema import ChannelConfig, ComputeConfig, ProjectConfig

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────

MODALITIES = ["fluorescence", "brightfield", "phase_contrast", "EM", "H&E", "confocal"]
STRUCTURES = ["nuclei", "whole_cells", "cytoplasm", "membrane", "organelles", "custom"]
IMAGE_FORMATS = ["tiff", "ome-tiff", "png", "jpg", "czi", "nd2", "lif"]
ANALYSIS_GOALS = ["count", "segment", "track", "measure", "classify"]
GROUND_TRUTH_FORMATS = ["masks", "points", "polygons"]

_FORMAT_EXTENSIONS: dict[str, list[str]] = {
    "tiff": [".tif", ".tiff"],
    "ome-tiff": [".ome.tif", ".ome.tiff"],
    "png": [".png"],
    "jpg": [".jpg", ".jpeg"],
    "czi": [".czi"],
    "nd2": [".nd2"],
    "lif": [".lif"],
}

# ── Document loading ───────────────────────────────────────────────────────────


def load_document(path: Path) -> str:
    """Read a document file to plain text.

    Supports .txt, .md, .rst, and .pdf (requires pypdf).

    Parameters
    ----------
    path:
        File to read.

    Returns
    -------
    Plain-text content of the document.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf  # type: ignore[import]

            reader = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files. Install it with: pip install pypdf"
            ) from None
    return path.read_text(encoding="utf-8", errors="replace")


# ── Document → field extraction ───────────────────────────────────────────────

_EXTRACTION_SCHEMA = """
Return ONLY a JSON object (no markdown fences) with these keys — omit any key you cannot determine:
{
  "name": "project name as a short string",
  "organism": "organism/species, e.g. human, mouse, zebrafish",
  "sample_type": "cell_culture | tissue_section | whole_mount | organoid",
  "modality": "fluorescence | brightfield | phase_contrast | EM | H&E | confocal",
  "structures": ["list", "of", "structures"],
  "channels": [{"name": "DAPI", "index": 0, "target": "nuclei"}],
  "image_format": "tiff | ome-tiff | png | jpg | czi | nd2 | lif",
  "bit_depth": 8 or 16,
  "has_ground_truth": true or false,
  "ground_truth_format": "masks | points | polygons",
  "analysis_goal": "count | segment | track | measure | classify",
  "compute": {"gpu_model": "...", "vram_gb": 8.0, "ram_gb": 32.0},
  "data_dir": "path/to/images",
  "gt_dir": "path/to/annotations"
}
"""


def _keyword_extract(text: str) -> dict[str, Any]:
    """Best-effort keyword extraction without an LLM."""
    result: dict[str, Any] = {}
    low = text.lower()

    # Organism
    for org in ("human", "mouse", "zebrafish", "drosophila", "plant", "rat", "yeast"):
        if org in low:
            result["organism"] = org
            break

    # Modality
    for mod in MODALITIES:
        if mod.lower() in low or mod.lower().replace("_", " ") in low:
            result["modality"] = mod
            break

    # Structures
    found_structures = [
        s for s in STRUCTURES if s.lower().replace("_", " ") in low or s.lower() in low
    ]
    if found_structures:
        result["structures"] = found_structures

    # Analysis goal
    for goal in ANALYSIS_GOALS:
        if goal in low:
            result["analysis_goal"] = goal
            break

    # Image format
    for fmt in IMAGE_FORMATS:
        if fmt in low or fmt.replace("-", " ") in low:
            result["image_format"] = fmt
            break
    # Also check extensions
    if "image_format" not in result:
        for fmt, exts in _FORMAT_EXTENSIONS.items():
            if any(ext.lstrip(".") in low for ext in exts):
                result["image_format"] = fmt
                break

    # Bit depth
    if "16-bit" in low or "16 bit" in low:
        result["bit_depth"] = 16
    elif "8-bit" in low or "8 bit" in low:
        result["bit_depth"] = 8

    # Ground truth
    if any(
        kw in low
        for kw in ("ground truth", "ground-truth", "annotation", "labeled", "labelled")
    ):
        result["has_ground_truth"] = True
        for gtf in GROUND_TRUTH_FORMATS:
            if gtf in low:
                result["ground_truth_format"] = gtf
                break

    # VRAM / RAM
    vram_m = re.search(r"(\d+(?:\.\d+)?)\s*gb?\s+(?:vram|gpu memory)", low)
    if vram_m:
        result.setdefault("compute", {})["vram_gb"] = float(vram_m.group(1))
    ram_m = re.search(r"(\d+(?:\.\d+)?)\s*gb?\s+(?:ram|memory)", low)
    if ram_m:
        result.setdefault("compute", {})["ram_gb"] = float(ram_m.group(1))

    return result


def extract_from_text(text: str) -> dict[str, Any]:
    """Extract ProjectConfig fields from free-form document text.

    Tries the Anthropic Claude API first (if ``anthropic`` is installed and
    ``ANTHROPIC_API_KEY`` is set). Falls back to keyword heuristics otherwise.

    Parameters
    ----------
    text:
        Raw document content.

    Returns
    -------
    Partial dict of project fields; keys present only when a value was found.
    """
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key:
        try:
            import anthropic  # type: ignore[import]

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract microscopy project information from the document below.\n\n"
                            f"{_EXTRACTION_SCHEMA}\n\n"
                            f"DOCUMENT:\n{text[:8000]}"
                        ),
                    }
                ],
            )
            raw_json = message.content[0].text.strip()
            # Strip optional markdown fences
            raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
            raw_json = re.sub(r"\s*```$", "", raw_json)
            extracted = json.loads(raw_json)
            return extracted
        except Exception as exc:
            console.print(
                f"[dim yellow]LLM extraction failed ({exc}); "
                "falling back to keyword search.[/dim yellow]"
            )

    return _keyword_extract(text)


# ── Auto-detection ─────────────────────────────────────────────────────────────


def auto_detect_from_directory(path: Path) -> dict[str, Any]:
    """Scan image directory and return detected properties as suggestions.

    Parameters
    ----------
    path:
        Directory to scan.

    Returns
    -------
    dict with keys: format, dimensions, dtype, channel_count (all may be None).
    """
    result: dict[str, Any] = {
        "format": None,
        "dimensions": None,
        "dtype": None,
        "channel_count": None,
    }

    candidates: list[Path] = []
    for exts in _FORMAT_EXTENSIONS.values():
        for ext in exts:
            candidates.extend(path.glob(f"*{ext}"))
            candidates.extend(path.glob(f"*{ext.upper()}"))
    if not candidates:
        return result

    from collections import Counter

    ext_counts: Counter = Counter()
    for f in candidates:
        ext_counts[f.suffix.lower()] += 1
    common_ext = ext_counts.most_common(1)[0][0]
    for fmt, exts in _FORMAT_EXTENSIONS.items():
        if common_ext in exts:
            result["format"] = fmt
            break

    first = candidates[0]
    try:
        if first.suffix.lower() in (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".png"):
            try:
                import tifffile

                img = tifffile.imread(str(first))
                result["dtype"] = str(img.dtype)
                if img.ndim == 2:
                    result["dimensions"] = (img.shape[0], img.shape[1])
                    result["channel_count"] = 1
                elif img.ndim == 3:
                    if img.shape[0] <= 8:
                        result["channel_count"] = img.shape[0]
                        result["dimensions"] = (img.shape[1], img.shape[2])
                    else:
                        result["channel_count"] = img.shape[2] if img.shape[2] <= 8 else 1
                        result["dimensions"] = (img.shape[0], img.shape[1])
            except Exception:
                pass
    except Exception:
        pass

    return result


# ── Model recommendation ───────────────────────────────────────────────────────


def recommend_model_from_properties(
    modality: str,
    structures: list[str],
    vram_gb: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Select the best segmentation model from normalized project properties.

    Decision matrix (Section 3.3):
    - EM / organelles → micro_sam vit_b_em_organelles
    - H&E → stardist 2D_versatile_he
    - Nuclei only → stardist 2D_versatile_fluo
    - phase_contrast / brightfield → cellpose cyto2
    - confocal + whole_cells → cellpose cpsam
    - fluorescence + whole_cells → cellpose cyto3
    - Low VRAM (<4 GB) → cellpose cyto2 (no GPU)
    - Default → cellpose cpsam
    """
    modality = modality.lower()
    structures = [s.lower() for s in structures]

    if modality == "em" or "organelles" in structures:
        return "micro_sam", {
            "model_type": "vit_b_em_organelles",
            "segmentation_mode": "auto",
            "min_size": 25,
        }

    if modality == "h&e":
        return "stardist", {
            "model_name": "2D_versatile_he",
            "prob_thresh": 0.5,
            "nms_thresh": 0.4,
        }

    nucleus_only = structures and all(s == "nuclei" for s in structures)
    if nucleus_only:
        return "stardist", {
            "model_name": "2D_versatile_fluo",
            "prob_thresh": 0.479071,
            "nms_thresh": 0.3,
        }

    if vram_gb is not None and vram_gb < 4.0:
        return "cellpose", {
            "model_name": "cyto2",
            "diameter": 30,
            "gpu": False,
            "flow_threshold": 0.4,
        }

    if modality in ("phase_contrast", "brightfield"):
        return "cellpose", {
            "model_name": "cyto2",
            "diameter": 30,
            "gpu": True,
            "flow_threshold": 0.4,
        }

    if modality == "confocal":
        return "cellpose", {
            "model_name": "cpsam",
            "diameter": 0,
            "gpu": True,
            "flow_threshold": 0.4,
        }

    if modality == "fluorescence":
        return "cellpose", {
            "model_name": "cyto3",
            "diameter": 0,
            "gpu": True,
            "flow_threshold": 0.4,
        }

    return "cellpose", {
        "model_name": "cpsam",
        "diameter": 0,
        "gpu": True,
        "flow_threshold": 0.4,
    }


def recommend_model(project: ProjectConfig) -> tuple[str, dict[str, Any]]:
    """Select the best segmentation model based on project properties."""
    return recommend_model_from_properties(
        modality=project.modality,
        structures=project.structures,
        vram_gb=project.compute.vram_gb,
    )


# ── YAML serialization ─────────────────────────────────────────────────────────


def _project_to_dict(project: ProjectConfig) -> dict:
    return {
        "name": project.name,
        "organism": project.organism,
        "sample_type": project.sample_type,
        "modality": project.modality,
        "structures": project.structures,
        "channels": [
            {"name": c.name, "index": c.index, "target": c.target}
            for c in project.channels
        ],
        "image_format": project.image_format,
        "typical_dimensions": (
            list(project.typical_dimensions) if project.typical_dimensions else None
        ),
        "bit_depth": project.bit_depth,
        "has_ground_truth": project.has_ground_truth,
        "ground_truth_format": project.ground_truth_format,
        "analysis_goal": project.analysis_goal,
        "compute": {
            "gpu_model": project.compute.gpu_model,
            "vram_gb": project.compute.vram_gb,
            "ram_gb": project.compute.ram_gb,
        },
        "data_dir": str(project.data_dir),
        "gt_dir": str(project.gt_dir) if project.gt_dir else None,
        "recommended_model": project.recommended_model,
        "recommended_params": project.recommended_params,
    }


def _dict_to_project(d: dict) -> ProjectConfig:
    channels = [
        ChannelConfig(name=c["name"], index=c["index"], target=c["target"])
        for c in d.get("channels", [])
    ]
    compute_d = d.get("compute") or {}
    compute = ComputeConfig(
        gpu_model=compute_d.get("gpu_model"),
        vram_gb=compute_d.get("vram_gb"),
        ram_gb=compute_d.get("ram_gb"),
    )
    dims_raw = d.get("typical_dimensions")
    typical_dimensions: tuple[int, int] | None = (
        (int(dims_raw[0]), int(dims_raw[1])) if dims_raw else None
    )
    gt_dir_raw = d.get("gt_dir")
    return ProjectConfig(
        name=d.get("name", "untitled"),
        organism=d.get("organism", "unknown"),
        sample_type=d.get("sample_type", "unknown"),
        modality=d.get("modality", "fluorescence"),
        structures=d.get("structures", []),
        channels=channels,
        image_format=d.get("image_format", "tiff"),
        typical_dimensions=typical_dimensions,
        bit_depth=int(d.get("bit_depth", 16)),
        has_ground_truth=bool(d.get("has_ground_truth", False)),
        ground_truth_format=d.get("ground_truth_format"),
        analysis_goal=d.get("analysis_goal", "segment"),
        compute=compute,
        data_dir=Path(d.get("data_dir", ".")),
        gt_dir=Path(gt_dir_raw) if gt_dir_raw else None,
        recommended_model=d.get("recommended_model", ""),
        recommended_params=d.get("recommended_params") or {},
    )


def save_project(project: ProjectConfig, path: Path = Path("project.yaml")) -> None:
    """Serialize ProjectConfig to YAML."""
    with open(path, "w") as fh:
        yaml.safe_dump(_project_to_dict(project), fh, default_flow_style=False, sort_keys=False)


def load_project(path: Path = Path("project.yaml")) -> ProjectConfig:
    """Load and validate project.yaml, filling defaults for missing fields."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _dict_to_project(raw)


# ── Interactive helpers ────────────────────────────────────────────────────────


def _prefilled(label: str, value: Any) -> None:
    """Print a line showing a field was pre-filled from the document."""
    console.print(f"  [dim]{label}:[/dim] [green]{value}[/green] [dim](from document)[/dim]")


def _ask_choice(
    prompt: str,
    choices: list[str],
    default: str | None = None,
    prefill: str | None = None,
) -> str:
    """Prompt user to choose from a numbered list, or confirm a pre-filled value."""
    if prefill and prefill in choices:
        _prefilled(prompt, prefill)
        override = Prompt.ask(
            f"  Accept [green]{prefill}[/green] or enter another number/value?",
            default="",
        )
        if not override:
            return prefill
        if override.isdigit():
            idx = int(override) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        if override in choices:
            return override
        console.print("[yellow]Invalid — keeping document value.[/yellow]")
        return prefill

    console.print(f"\n[bold]{prompt}[/bold]")
    for i, choice in enumerate(choices, 1):
        marker = " [dim](default)[/dim]" if choice == default else ""
        console.print(f"  {i}. {choice}{marker}")
    while True:
        raw = Prompt.ask("Enter number or value", default=default or "")
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        if raw in choices:
            return raw
        if raw == "" and default:
            return default
        console.print(
            f"[yellow]Please enter a number 1-{len(choices)} "
            f"or one of: {', '.join(choices)}[/yellow]"
        )


def _ask_multiselect(
    prompt: str,
    choices: list[str],
    prefill: list[str] | None = None,
) -> list[str]:
    """Prompt user to select one or more items, or confirm pre-filled values."""
    if prefill:
        valid_prefill = [s for s in prefill if s in choices]
        if valid_prefill:
            _prefilled(prompt, ", ".join(valid_prefill))
            keep = Confirm.ask("  Accept these structures?", default=True)
            if keep:
                return valid_prefill

    console.print(f"\n[bold]{prompt}[/bold]")
    for i, choice in enumerate(choices, 1):
        console.print(f"  {i}. {choice}")
    console.print("  Enter numbers separated by commas, e.g. 1,3")
    while True:
        raw = Prompt.ask("Selection")
        parts = [p.strip() for p in raw.split(",")]
        selected: list[str] = []
        valid = True
        for part in parts:
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(choices):
                    selected.append(choices[idx])
                    continue
            if part in choices:
                selected.append(part)
                continue
            console.print(f"[yellow]Unknown option: {part!r}[/yellow]")
            valid = False
            break
        if valid and selected:
            return selected
        console.print(
            f"[yellow]Please enter comma-separated numbers 1-{len(choices)} "
            "or option names.[/yellow]"
        )


def _read_pasted_text() -> str:
    """Read multi-line text pasted by the user until they enter a blank line."""
    console.print(
        "[dim]Paste your document below and press [bold]Enter twice[/bold] "
        "(blank line) when done:[/dim]"
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines)


# ── Main interview ─────────────────────────────────────────────────────────────


def create_project_interactive(
    data_dir: Path | None = None,
    prefill: dict[str, Any] | None = None,
) -> ProjectConfig:
    """Run an interactive interview, skipping questions already answered in *prefill*.

    Parameters
    ----------
    data_dir:
        Optional image directory for auto-detection hints.
    prefill:
        Dict of pre-extracted fields (from a document or LLM). Any field
        present here is shown as "from document" and the user can confirm or
        override it; questions for absent fields are asked normally.

    Returns
    -------
    ProjectConfig with recommended_model and recommended_params filled.
    """
    console.print(Panel("[bold cyan]MicroAgent Project Setup[/bold cyan]", border_style="cyan"))
    p = prefill or {}

    # Auto-detect from image directory
    hints: dict[str, Any] = {}
    if data_dir and data_dir.is_dir():
        with console.status(f"[dim]Scanning {data_dir} …[/dim]"):
            hints = auto_detect_from_directory(data_dir)
        detected = ", ".join(f"{k}={v}" for k, v in hints.items() if v is not None)
        if detected:
            console.print(f"[dim]Auto-detected from images: {detected}[/dim]")

    # Count how many questions can be skipped
    pre_count = sum(
        1 for k in ("name", "organism", "modality", "structures", "analysis_goal") if k in p
    )
    if pre_count:
        console.print(
            f"[green]{pre_count} field(s) pre-filled from document[/green] — "
            "press Enter to accept each or type a new value.\n"
        )

    # ── Q1: Project name ──────────────────────────────────────────────────────
    if "name" in p:
        _prefilled("Project name", p["name"])
        override = Prompt.ask("  Accept or enter another name?", default="")
        name = override if override else p["name"]
    else:
        name = Prompt.ask("\n[bold]What is your project name?[/bold]")

    # ── Q2: Organism ──────────────────────────────────────────────────────────
    if "organism" in p:
        _prefilled("Organism", p["organism"])
        override = Prompt.ask("  Accept or enter another organism?", default="")
        organism = override if override else p["organism"]
    else:
        organism = Prompt.ask(
            "\n[bold]What organism / sample?[/bold] (e.g. human, mouse, zebrafish)",
            default="human",
        )

    # ── Q3: Modality ──────────────────────────────────────────────────────────
    modality = _ask_choice(
        "What microscopy modality?",
        MODALITIES,
        default="fluorescence",
        prefill=p.get("modality"),
    )

    # ── Q4: Structures ────────────────────────────────────────────────────────
    structures = _ask_multiselect(
        "What structures do you want to segment?",
        STRUCTURES,
        prefill=p.get("structures"),
    )

    # ── Q5: Channels ──────────────────────────────────────────────────────────
    prefill_channels: list[dict] = p.get("channels", [])
    if prefill_channels:
        console.print(
            f"\n  [dim]Channels:[/dim] [green]{prefill_channels}[/green] "
            "[dim](from document)[/dim]"
        )
        keep_ch = Confirm.ask("  Accept these channels?", default=True)
        if keep_ch:
            channels = [
                ChannelConfig(name=c["name"], index=c["index"], target=c["target"])
                for c in prefill_channels
            ]
        else:
            prefill_channels = []

    if not prefill_channels:
        suggested_ch = hints.get("channel_count") or 1
        console.print(f"\n[bold]Describe your channel setup.[/bold]  (detected: {suggested_ch})")
        n_channels = IntPrompt.ask("  How many channels?", default=suggested_ch)
        channels = []
        for i in range(n_channels):
            ch_name = Prompt.ask(f"  Channel {i} name (e.g. DAPI, GFP)", default=f"ch{i}")
            ch_target = Prompt.ask(
                f"  Channel {i} images what structure?",
                default=structures[0] if structures else "nuclei",
            )
            channels.append(ChannelConfig(name=ch_name, index=i, target=ch_target))

    # ── Q6: Image format ──────────────────────────────────────────────────────
    fmt_default = hints.get("format") or "tiff"
    image_format = _ask_choice(
        "What image format?",
        IMAGE_FORMATS,
        default=fmt_default,
        prefill=p.get("image_format"),
    )

    # ── Q7: Ground truth ──────────────────────────────────────────────────────
    gt_prefill = p.get("has_ground_truth")
    if gt_prefill is not None:
        _prefilled("Has ground truth", gt_prefill)
        has_gt = Confirm.ask("  Accept?", default=bool(gt_prefill))
    else:
        has_gt = Confirm.ask("\n[bold]Do you have ground-truth annotations?[/bold]", default=False)

    gt_format: str | None = None
    gt_dir: Path | None = None
    if has_gt:
        gt_format = _ask_choice(
            "Ground truth format?",
            GROUND_TRUTH_FORMATS,
            default="masks",
            prefill=p.get("ground_truth_format"),
        )
        gt_dir_default = str(p.get("gt_dir", "annotations"))
        gt_dir_str = Prompt.ask("  Path to ground-truth directory", default=gt_dir_default)
        gt_dir = Path(gt_dir_str)

    # ── Q8: Analysis goal ─────────────────────────────────────────────────────
    analysis_goal = _ask_choice(
        "What is your analysis goal?",
        ANALYSIS_GOALS,
        default="segment",
        prefill=p.get("analysis_goal"),
    )

    # ── Q9: Compute ───────────────────────────────────────────────────────────
    prefill_compute = p.get("compute") or {}
    console.print("\n[bold]Compute environment[/bold] (press Enter to skip each)")
    gpu_default = prefill_compute.get("gpu_model") or ""
    vram_default = (
        str(prefill_compute.get("vram_gb", "")) if prefill_compute.get("vram_gb") else ""
    )
    ram_default = str(prefill_compute.get("ram_gb", "")) if prefill_compute.get("ram_gb") else ""
    if gpu_default:
        console.print(f"  [dim]GPU model from document:[/dim] [green]{gpu_default}[/green]")
    gpu_model_raw = Prompt.ask("  GPU model (e.g. RTX 3090)", default=gpu_default)
    vram_raw = Prompt.ask("  GPU VRAM in GB", default=vram_default)
    ram_raw = Prompt.ask("  System RAM in GB", default=ram_default)

    compute = ComputeConfig(
        gpu_model=gpu_model_raw or None,
        vram_gb=float(vram_raw) if vram_raw else None,
        ram_gb=float(ram_raw) if ram_raw else None,
    )

    # ── Assemble config ───────────────────────────────────────────────────────
    dims = hints.get("dimensions")
    typical_dimensions: tuple[int, int] | None = tuple(dims) if dims else None  # type: ignore[assignment]
    dtype_str = hints.get("dtype") or ""
    bit_depth = p.get("bit_depth") or (16 if "16" in dtype_str else 8)

    project = ProjectConfig(
        name=name,
        organism=organism,
        sample_type=p.get("sample_type", "cell_culture"),
        modality=modality,
        structures=structures,
        channels=channels,
        image_format=image_format,
        typical_dimensions=typical_dimensions,
        bit_depth=int(bit_depth),
        has_ground_truth=has_gt,
        ground_truth_format=gt_format,
        analysis_goal=analysis_goal,
        compute=compute,
        data_dir=data_dir or Path(p.get("data_dir", ".")),
        gt_dir=gt_dir,
    )

    rec_model, rec_params = recommend_model(project)
    project.recommended_model = rec_model
    project.recommended_params = rec_params

    console.print(
        Panel(
            f"[bold]Model:[/bold]  {rec_model}\n"
            + "\n".join(f"  [dim]{k}:[/dim] {v}" for k, v in rec_params.items()),
            title="Recommended Model",
            border_style="green",
        )
    )
    return project
