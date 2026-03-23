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


@app.command()
def train(
    image_dir: Path = typer.Argument(..., help="Directory containing raw training images"),
    gt_dir: Path = typer.Argument(..., help="Directory containing ground-truth masks"),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    weight_decay: float = typer.Option(0.1, "--weight-decay", help="Weight decay"),
    pretrained: str = typer.Option("cpsam", "--pretrained", help="Pretrained model name"),
    test_split: float = typer.Option(0.2, "--test-split", help="Fraction reserved for test set"),
    output: Path = typer.Option(Path("models"), "--output", "-o", help="Directory to save trained model"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Fine-tune a CellPose model on labelled microscopy images."""
    from microagent.core.train import TrainConfig, prepare_data, train_cellpose

    # ── Prepare data ──────────────────────────────────────────────────────────
    with console.status("[bold green]Preparing training data …"):
        try:
            train_dir, test_dir = prepare_data(
                image_dir=image_dir,
                gt_dir=gt_dir,
                test_split=test_split,
                seed=seed,
            )
        except (ValueError, RuntimeError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from None

    console.print(
        f"[green]✓ Data prepared[/green]  "
        f"train={train_dir}  test={test_dir}"
    )

    # ── Build config ──────────────────────────────────────────────────────────
    config = TrainConfig(
        pretrained=pretrained,
        train_dir=train_dir,
        test_dir=test_dir,
        learning_rate=lr,
        weight_decay=weight_decay,
        n_epochs=epochs,
        seed=seed,
        save_dir=output,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    from rich.live import Live
    from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn
    from rich.table import Table

    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task("Training …", total=epochs)

    try:
        with progress:
            # CellPose training is blocking; we update progress after it
            # completes. For interactive epoch-level feedback, users can
            # monitor the rich console output from cellpose itself.
            result = train_cellpose(config)
            progress.update(task_id, completed=epochs)
    except ImportError as exc:
        console.print(f"[bold red]Import error:[/bold red] {exc}")
        raise typer.Exit(1) from None
    except (ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Training error:[/bold red] {exc}")
        raise typer.Exit(1) from None

    # ── Summary ───────────────────────────────────────────────────────────────
    tbl = Table(title="Training Summary", box=SIMPLE_HEAD, show_lines=False)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Best epoch", str(result.best_epoch + 1))
    tbl.add_row("Epochs run", str(len(result.train_losses) or epochs))
    if result.train_losses:
        tbl.add_row("Final train loss", f"{result.train_losses[-1]:.4f}")
    if result.test_losses:
        tbl.add_row("Final test loss", f"{result.test_losses[-1]:.4f}")
        tbl.add_row("Best test loss", f"{min(result.test_losses):.4f}")
    tbl.add_row("Elapsed (s)", f"{result.elapsed_seconds:.1f}")
    console.print(tbl)
    console.print(f"\n[green]✓ Model saved →[/green] {result.model_path}")


@app.command()
def evaluate(
    pred_dir: Path = typer.Argument(..., help="Directory containing predicted mask TIFFs"),
    gt_dir: Path = typer.Argument(..., help="Directory containing ground-truth mask TIFFs"),
    thresholds: str = typer.Option(
        "0.5,0.75,0.9",
        "--thresholds",
        help="Comma-separated IoU thresholds",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save metrics JSON to this path",
    ),
    compare: Optional[Path] = typer.Option(
        None,
        "--compare",
        help="Compare against a previously saved metrics JSON",
    ),
) -> None:
    """Evaluate predicted masks against ground-truth and print metrics."""
    from microagent.core.evaluate import compare_runs, evaluate_masks

    try:
        thresh_list = [float(t.strip()) for t in thresholds.split(",")]
    except ValueError:
        console.print(f"[bold red]Invalid --thresholds value:[/bold red] {thresholds}")
        raise typer.Exit(1)

    with console.status("[bold green]Evaluating masks …"):
        result = evaluate_masks(pred_dir, gt_dir, thresholds=thresh_list)

    if compare and compare.exists():
        from microagent.core.evaluate import EvaluationResult

        baseline = EvaluationResult.load_json(compare)
        result.comparison = compare_runs(baseline, result)

    # ── Warnings for unmatched files ──────────────────────────────────────────
    if result.unmatched_preds or result.unmatched_gts:
        lines: list[str] = []
        for f in result.unmatched_preds:
            lines.append(f"  [yellow]Unmatched prediction:[/yellow] {f}")
        for f in result.unmatched_gts:
            lines.append(f"  [yellow]Unmatched ground-truth:[/yellow] {f}")
        console.print(Panel("\n".join(lines), title="Unmatched Files", border_style="yellow"))

    # ── Per-image metrics table ───────────────────────────────────────────────
    def _colour(v: float) -> str:
        if v >= 0.8:
            return "green"
        if v >= 0.5:
            return "yellow"
        return "red"

    def _fmt(v: float) -> Text:
        return Text(f"{v:.3f}", style=_colour(v))

    tbl = Table(title="Per-Image Metrics", box=SIMPLE_HEAD, show_lines=False)
    tbl.add_column("Filename", style="cyan", no_wrap=True)
    tbl.add_column("GT", justify="right")
    tbl.add_column("Pred", justify="right")
    for t in thresh_list:
        tbl.add_column(f"F1@{t}", justify="right")
    tbl.add_column("mAP", justify="right")
    tbl.add_column("PQ", justify="right")

    for im in result.per_image:
        row: list[str | Text] = [
            im.filename,
            str(im.gt_count),
            str(im.pred_count),
        ]
        for tm in im.per_threshold:
            row.append(_fmt(tm.f1))
        row.append(_fmt(im.map))
        row.append(_fmt(im.panoptic_quality))
        tbl.add_row(*row)

    # Summary row
    s = result.summary
    summary_row: list[str | Text] = [
        "[bold]SUMMARY[/bold]",
        f"{s.mean_gt_count:.1f}",
        f"{s.mean_pred_count:.1f}",
    ]
    for tm in s.per_threshold:
        summary_row.append(_fmt(tm.f1))
    summary_row.append(_fmt(s.map))
    summary_row.append(_fmt(s.panoptic_quality))
    tbl.add_row(*summary_row)

    console.print(tbl)

    # ── Worst-image callout ───────────────────────────────────────────────────
    if result.worst_images:
        worst_lines = "\n".join(f"  [red]•[/red] {f}" for f in result.worst_images)
        console.print(Panel(worst_lines, title="Worst Images (lowest F1@0.5)", border_style="red"))

    # ── Comparison ───────────────────────────────────────────────────────────
    if result.comparison:
        c = result.comparison
        delta_lines: list[str] = []
        for metric, delta in c.metric_deltas.items():
            arrow = "↑" if delta > 1e-9 else ("↓" if delta < -1e-9 else "→")
            colour = "green" if delta > 1e-9 else ("red" if delta < -1e-9 else "dim")
            delta_lines.append(f"  [{colour}]{arrow} {metric}: {delta:+.4f}[/{colour}]")
        if c.improved_images:
            delta_lines.append(f"\n  [green]Improved:[/green] {', '.join(c.improved_images)}")
        if c.regressed_images:
            delta_lines.append(f"  [red]Regressed:[/red] {', '.join(c.regressed_images)}")
        console.print(Panel("\n".join(delta_lines), title="Comparison vs Baseline", border_style="blue"))

    # ── JSON output ───────────────────────────────────────────────────────────
    if output:
        result.save_json(output)
        console.print(f"\n[green]Metrics saved →[/green] {output}")
