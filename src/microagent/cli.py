from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="microagent",
    help="Agentic microscopy image analysis tool.",
    add_completion=False,
)

console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def inspect(
    directory: Path = typer.Argument(..., help="Directory containing images to inspect"),
    channels: Optional[str] = typer.Option(
        None,
        "--channels",
        help="Comma-separated channel indices to analyse, e.g. 0,1",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save JSON report to this path",
    ),
) -> None:
    """Inspect a directory of microscopy images and run QC checks."""
    from microagent.core.inspect import inspect_directory

    ch_list: list[int] | None = None
    if channels:
        try:
            ch_list = [int(c.strip()) for c in channels.split(",")]
        except ValueError:
            console.print(f"[bold red]Invalid --channels value:[/bold red] {channels}")
            raise typer.Exit(1)

    with console.status(f"[bold green]Inspecting {directory} …"):
        report = inspect_directory(directory, channels=ch_list)

    # ── Files table ──────────────────────────────────────────────────────────
    tbl = Table(title="Image Files", box=SIMPLE_HEAD, show_lines=False)
    tbl.add_column("#", style="dim", width=5, justify="right")
    tbl.add_column("Filename", style="cyan", no_wrap=True)
    for i, fp in enumerate(report.file_paths, 1):
        tbl.add_row(str(i), Path(fp).name)
    console.print(tbl)

    # ── Stats panel ──────────────────────────────────────────────────────────
    stats_lines: list[str] = [
        f"[bold]Files loaded:[/bold]    {report.file_count}",
        f"[bold]Channels:[/bold]        {report.channel_count}",
        f"[bold]Dtypes:[/bold]          {', '.join(report.dtypes) or '—'}",
        f"[bold]Dimensions:[/bold]      {', '.join(str(d) for d in report.dimensions) or '—'}",
    ]
    if report.intensity_stats:
        stats_lines.append("")
        stats_lines.append("[bold]Intensity statistics (all images):[/bold]")
        for cs in report.intensity_stats:
            stats_lines.append(
                f"  ch{cs.channel}  min={cs.min:.1f}  max={cs.max:.1f}"
                f"  mean={cs.mean:.1f}  std={cs.std:.1f}"
            )
    if report.thumbnail_paths:
        stats_lines.append("")
        stats_lines.append(f"[bold]Thumbnail:[/bold] {report.thumbnail_paths[0]}")

    console.print(Panel("\n".join(stats_lines), title="Summary", border_style="green"))

    # ── QC issues ────────────────────────────────────────────────────────────
    if report.issues:
        issue_text = Text()
        for issue in report.issues:
            # Errors (load failures) in red, warnings in yellow
            colour = "red" if issue.startswith("Failed") else "yellow"
            issue_text.append(f"  • {issue}\n", style=colour)
        console.print(Panel(issue_text, title="QC Warnings", border_style="yellow"))
    else:
        console.print("[bold green]✓ No QC issues detected[/bold green]")

    # ── JSON output ──────────────────────────────────────────────────────────
    if output:
        report.save_json(output)
        console.print(f"\n[green]Report saved →[/green] {output}")


@app.command()
def segment(
    image_dir: Path = typer.Argument(..., help="Directory containing images to segment"),
    output: Path = typer.Option(
        Path("masks"),
        "--output",
        "-o",
        help="Directory to write mask TIFFs into",
    ),
    model: str = typer.Option(
        "auto",
        "--model",
        "-m",
        help="Model backend: auto, cellpose, or stardist",
    ),
    diameter: int | None = typer.Option(
        None,
        "--diameter",
        help="Expected cell diameter in pixels (CellPose only)",
    ),
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to project.yaml for model auto-selection and defaults",
    ),
) -> None:
    """Segment images and save labeled TIFF masks."""
    from microagent.core.segment import run_segmentation

    kwargs: dict = {}
    if diameter is not None:
        kwargs["diameter"] = diameter

    output.mkdir(parents=True, exist_ok=True)

    with console.status(f"[bold green]Segmenting images in {image_dir} …"):
        try:
            result = run_segmentation(
                image_dir=image_dir,
                output_dir=output,
                model=model,
                project_path=project,
                **kwargs,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from None

    # ── Model info panel ──────────────────────────────────────────────────────
    info = result.model_info
    console.print(
        Panel(
            f"[bold]Backend:[/bold]  {info['backend']}\n"
            f"[bold]Model:[/bold]    {info['model_name']}\n"
            f"[bold]Params:[/bold]   {info['parameters']}",
            title="Model",
            border_style="blue",
        )
    )

    # ── Per-image results table ───────────────────────────────────────────────
    tbl = Table(title="Segmentation Results", box=SIMPLE_HEAD, show_lines=False)
    tbl.add_column("#", style="dim", width=5, justify="right")
    tbl.add_column("Filename", style="cyan", no_wrap=True)
    tbl.add_column("Cells found", justify="right")
    tbl.add_column("Time (s)", justify="right")
    for i, stat in enumerate(result.per_image_stats, 1):
        tbl.add_row(
            str(i),
            stat.filename,
            str(stat.n_labels),
            f"{stat.elapsed_seconds:.2f}",
        )
    console.print(tbl)

    console.print(
        f"\n[green]✓ {len(result.mask_paths)} masks saved →[/green] {output}  "
        f"[dim](total {result.elapsed_seconds:.1f}s)[/dim]"
    )
