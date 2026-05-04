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
def optimize(
    image_dir: Path = typer.Argument(..., help="Directory containing images"),
    gt_dir: Path = typer.Argument(..., help="Directory containing ground-truth masks"),
    trials: int = typer.Option(20, "--trials", "-n", help="Number of Optuna trials"),
    metric: str = typer.Option("f1", "--metric", help="Metric to optimise: f1, map, pq, precision, recall"),
    model: str = typer.Option("auto", "--model", "-m", help="Backend: auto, cellpose, or stardist"),
    iou: float = typer.Option(0.5, "--iou", help="IoU threshold for F1/precision/recall"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    project: Path | None = typer.Option(None, "--project", "-p", help="Path to project.yaml"),
) -> None:
    """Optimise segmentation hyperparameters with Optuna TPE search."""
    from rich.live import Live
    from rich.table import Table

    from microagent.core.optimize import OptimizeConfig, run_optimization

    config = OptimizeConfig(
        image_dir=image_dir,
        gt_dir=gt_dir,
        model=model,
        n_trials=trials,
        metric=metric,
        iou_threshold=iou,
        seed=seed,
        project_path=project,
    )

    # ── Live trial table ──────────────────────────────────────────────────────
    def _make_table(records: list) -> Table:
        tbl = Table(title="Optuna Trials", box=SIMPLE_HEAD, show_lines=False)
        tbl.add_column("Trial", style="dim", width=6, justify="right")
        tbl.add_column("Params", style="cyan")
        tbl.add_column(metric.upper(), justify="right")
        tbl.add_column("Best so far", justify="right")
        best_so_far = 0.0
        for rec in records:
            if rec.value > best_so_far:
                best_so_far = rec.value
            param_str = "  ".join(f"{k}={v:.3g}" for k, v in rec.params.items())
            score_style = "green" if rec.value >= 0.7 else ("yellow" if rec.value >= 0.4 else "red")
            tbl.add_row(
                str(rec.number),
                param_str,
                f"[{score_style}]{rec.value:.4f}[/{score_style}]",
                f"{best_so_far:.4f}",
            )
        return tbl

    records: list = []

    with Live(console=console, refresh_per_second=4) as live:
        live.update(_make_table(records))

        def _on_trial(record) -> None:
            records.append(record)
            live.update(_make_table(records))

        try:
            result = run_optimization(config, on_trial_complete=_on_trial)
        except ImportError as exc:
            console.print(f"[bold red]Import error:[/bold red] {exc}")
            raise typer.Exit(1) from None
        except (FileNotFoundError, RuntimeError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from None

    # ── Final summary ─────────────────────────────────────────────────────────
    delta = result.improvement
    delta_colour = "green" if delta > 1e-6 else ("red" if delta < -1e-6 else "dim")
    param_lines = "\n".join(
        f"  [bold]{k}:[/bold] {v:.4g}" for k, v in result.best_params.items()
    )
    summary = (
        f"{param_lines}\n\n"
        f"[bold]Baseline {metric.upper()}:[/bold]  {result.baseline_value:.4f}\n"
        f"[bold]Best {metric.upper()}:[/bold]      {result.best_value:.4f}\n"
        f"[bold]Improvement:[/bold]    [{delta_colour}]{delta:+.4f}[/{delta_colour}]"
    )
    console.print(
        Panel(summary, title="Best Hyperparameters", border_style="green")
    )
    if result.study_path:
        console.print(f"[dim]Study saved → {result.study_path}[/dim]")


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


@app.command()
def init(
    data_dir: Path = typer.Option(
        Path("images"),
        "--data-dir",
        help="Directory containing your microscopy images",
    ),
    doc: Optional[Path] = typer.Option(
        None,
        "--doc",
        "-d",
        help="Project document to extract fields from (markdown, text, or PDF). "
        "Fields found in the document are pre-filled; only missing ones are asked.",
    ),
    output: Path = typer.Option(
        Path("project.yaml"),
        "--output",
        "-o",
        help="Where to save the generated project.yaml",
    ),
) -> None:
    """Run the interactive project setup interview and generate project.yaml.

    If --doc is given, fields are extracted from the document first and only
    missing information is asked interactively.  During the interview you can
    also paste a document when prompted.
    """
    from microagent.project.knowledge import (
        create_project_interactive,
        extract_from_text,
        load_document,
        save_project,
    )

    prefill: dict = {}

    # ── Load document from --doc flag ─────────────────────────────────────────
    if doc is not None:
        if not doc.exists():
            console.print(f"[bold red]Document not found:[/bold red] {doc}")
            raise typer.Exit(1)
        with console.status(f"[bold green]Reading {doc} …"):
            try:
                text = load_document(doc)
            except ImportError as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
                raise typer.Exit(1) from None
        with console.status("[bold green]Extracting project fields …"):
            prefill = extract_from_text(text)
        if prefill:
            console.print(
                f"[green]Extracted {len(prefill)} field(s) from {doc.name}.[/green]"
            )
        else:
            console.print("[yellow]No fields could be extracted from the document.[/yellow]")

    # ── Offer paste option when no --doc was given ────────────────────────────
    else:
        console.print(
            "[dim]Tip: if you have a project description document, pass it with "
            "[bold]--doc path/to/file[/bold] to pre-fill answers.[/dim]"
        )
        want_paste = typer.confirm(
            "Do you want to paste a project document now?", default=False
        )
        if want_paste:
            from microagent.project.knowledge import _read_pasted_text

            text = _read_pasted_text()
            if text.strip():
                with console.status("[bold green]Extracting project fields …"):
                    prefill = extract_from_text(text)
                if prefill:
                    console.print(
                        f"[green]Extracted {len(prefill)} field(s) from pasted text.[/green]"
                    )

    project = create_project_interactive(
        data_dir=data_dir if data_dir.exists() else None,
        prefill=prefill or None,
    )
    save_project(project, output)
    console.print(f"\n[green]✓ Project saved →[/green] {output}")
    console.print(
        f"[dim]Recommended model: {project.recommended_model}  "
        f"params: {project.recommended_params}[/dim]"
    )


@app.command()
def demo(
    output: Path = typer.Option(
        Path("demo_output"),
        "--output",
        "-o",
        help="Directory to write all demo outputs into",
    ),
    n_images: int = typer.Option(
        10,
        "--n-images",
        help="Number of synthetic images to generate",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Skip opening the report in a browser when done",
    ),
) -> None:
    """Generate synthetic data and run the full pipeline as a demo.

    Produces a self-contained HTML report without any real data or GPU.
    Target runtime: < 60 seconds on a machine with GPU, longer on CPU-only.
    """
    from microagent.core.evaluate import evaluate_masks
    from microagent.core.inspect import inspect_directory
    from microagent.core.segment import run_segmentation
    from microagent.demo.synthetic import generate_synthetic_dataset
    from microagent.viz.overlays import create_overlay, save_overlay_montage
    from microagent.viz.plots import plot_metrics_summary, plot_object_size_distribution
    from microagent.viz.report import generate_report, load_report_data

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate synthetic dataset ─────────────────────────────────────
    console.print(Panel(
        "[bold]MicroAgent Demo[/bold]\n"
        f"Generating {n_images} synthetic fluorescence images → {output}",
        border_style="blue",
    ))
    with console.status("[bold green]Generating synthetic microscopy data …"):
        image_dir, gt_dir = generate_synthetic_dataset(
            output,
            n_images=n_images,
            image_size=(512, 512),
            n_objects_range=(5, 25),
            noise_level=0.08,
            seed=42,
        )
    console.print(f"[green]✓ {n_images} images → {image_dir}[/green]")

    project_yaml = output / "project.yaml"

    # ── 2. Inspect ────────────────────────────────────────────────────────
    with console.status("[bold green]Running QC inspection …"):
        insp = inspect_directory(image_dir)
    insp_json = output / "inspection.json"
    insp.save_json(insp_json)
    console.print(
        f"[green]✓ Inspect:[/green] {insp.file_count} images, "
        f"{insp.channel_count} channels, {len(insp.issues)} QC issues"
    )

    # ── 3. Segment ────────────────────────────────────────────────────────
    masks_dir = output / "masks"
    masks_dir.mkdir(exist_ok=True)
    with console.status("[bold green]Segmenting nuclei with CellPose …"):
        try:
            seg = run_segmentation(
                image_dir=image_dir,
                output_dir=masks_dir,
                model="cellpose",
                project_path=project_yaml if project_yaml.exists() else None,
            )
        except (ImportError, RuntimeError) as exc:
            console.print(f"[bold red]Segmentation error:[/bold red] {exc}")
            raise typer.Exit(1) from None
    seg_json = output / "segmentation.json"
    seg.save_json(seg_json)
    total_cells = sum(s.n_labels for s in seg.per_image_stats)
    console.print(
        f"[green]✓ Segment:[/green] {len(seg.mask_paths)} masks, "
        f"{total_cells} total objects detected ({seg.elapsed_seconds:.1f}s)"
    )

    # ── 4. Generate overlay montage ───────────────────────────────────────
    overlays_dir = output / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    try:
        import tifffile

        imgs, masks_list = [], []
        for mask_path_str in seg.mask_paths[:9]:
            mask_path = Path(mask_path_str)
            img_name = mask_path.stem.replace("_mask", "") + ".tif"
            img_path = image_dir / img_name
            if img_path.exists():
                img_arr = tifffile.imread(str(img_path))
                mask_arr = tifffile.imread(str(mask_path))
                imgs.append(img_arr)
                masks_list.append(mask_arr)

        if imgs:
            montage_path = overlays_dir / "overlay_montage.png"
            save_overlay_montage(imgs, masks_list, montage_path, ncols=min(3, len(imgs)))
            console.print(f"[green]✓ Overlays → {montage_path}[/green]")
    except Exception as exc:
        console.print(f"[yellow]Overlay generation skipped: {exc}[/yellow]")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    with console.status("[bold green]Evaluating segmentation quality …"):
        ev = evaluate_masks(
            pred_dir=masks_dir,
            gt_dir=gt_dir,
            thresholds=[0.5, 0.75, 0.9],
            force_fallback=True,
        )
    ev_json = output / "metrics.json"
    ev.save_json(ev_json)
    f1_05 = next(
        (tm.f1 for tm in ev.summary.per_threshold if abs(tm.threshold - 0.5) < 1e-9),
        0.0,
    )
    console.print(
        f"[green]✓ Evaluate:[/green] F1@0.5={f1_05:.3f}  "
        f"mAP={ev.summary.map:.3f}  PQ={ev.summary.panoptic_quality:.3f}"
    )

    # ── 6. Generate metric plots ──────────────────────────────────────────
    plots_dir = output / "plots"
    plots_dir.mkdir(exist_ok=True)
    try:
        plot_metrics_summary(ev, plots_dir / "metrics_summary.png")
        # Aggregate all masks for size distribution
        import tifffile as _tf
        import numpy as np

        combined = np.concatenate([
            _tf.imread(str(p)).ravel() for p in masks_dir.glob("*_mask.tif")
        ])
        # Re-label from combined to get proper regionprops
        from skimage import measure as _measure
        combined_2d = _tf.imread(str(next(masks_dir.glob("*_mask.tif"))))
        plot_object_size_distribution(combined_2d, plots_dir / "object_sizes.png")
        console.print(f"[green]✓ Plots → {plots_dir}[/green]")
    except Exception as exc:
        console.print(f"[yellow]Plot generation skipped: {exc}[/yellow]")

    # ── 7. Generate HTML report ───────────────────────────────────────────
    report_path = output / "report.html"
    with console.status("[bold green]Building HTML report …"):
        try:
            data = load_report_data(
                inspection_json=insp_json,
                segmentation_json=seg_json,
                evaluation_json=ev_json,
                project_yaml=project_yaml if project_yaml.exists() else None,
                overlay_dir=overlays_dir if overlays_dir.is_dir() else None,
                plots_dir=plots_dir if plots_dir.is_dir() else None,
                command="microagent demo",
            )
            generate_report(data, report_path)
        except Exception as exc:
            console.print(f"[bold red]Report generation failed:[/bold red] {exc}")
            raise typer.Exit(1) from None

    console.print(
        Panel(
            f"[bold green]Demo complete![/bold green]\n\n"
            f"  Images:    {image_dir}\n"
            f"  Masks:     {masks_dir}\n"
            f"  Metrics:   {ev_json}\n"
            f"  Report:    {report_path}\n\n"
            f"  F1@0.5 = {f1_05:.3f}",
            title="Summary",
            border_style="green",
        )
    )

    if not no_browser:
        import webbrowser

        url = report_path.resolve().as_uri()
        console.print(f"[dim]Opening {url} …[/dim]")
        webbrowser.open(url)


@app.command()
def report(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to project.yaml",
    ),
    output: Path = typer.Option(
        Path("report.html"),
        "--output",
        "-o",
        help="Destination HTML file",
    ),
    inspection_json: Optional[Path] = typer.Option(
        None,
        "--inspection",
        help="Path to inspection JSON (auto-detected as inspection.json if omitted)",
    ),
    segmentation_json: Optional[Path] = typer.Option(
        None,
        "--segmentation",
        help="Path to segmentation JSON (auto-detected as segmentation.json if omitted)",
    ),
    evaluation_json: Optional[Path] = typer.Option(
        None,
        "--evaluation",
        help="Path to evaluation/metrics JSON (auto-detected as metrics.json if omitted)",
    ),
    optimization_json: Optional[Path] = typer.Option(
        None,
        "--optimization",
        help="Path to optimization JSON (auto-detected as optimization.json if omitted)",
    ),
    overlay_dir: Optional[Path] = typer.Option(
        None,
        "--overlays",
        help="Directory of overlay PNGs (auto-detected as overlays/ if omitted)",
    ),
    plots_dir: Optional[Path] = typer.Option(
        None,
        "--plots",
        help="Directory of metric plot PNGs (auto-detected as plots/ if omitted)",
    ),
) -> None:
    """Generate a self-contained HTML report from pipeline results."""
    from microagent.viz.report import generate_report, load_report_data

    # ── Auto-discover result files if not provided ────────────────────────────
    def _auto(explicit: Optional[Path], *candidates: str) -> Optional[Path]:
        if explicit:
            return explicit
        for name in candidates:
            p = Path(name)
            if p.exists():
                return p
        return None

    resolved_inspection = _auto(inspection_json, "inspection.json")
    resolved_segmentation = _auto(segmentation_json, "segmentation.json")
    resolved_evaluation = _auto(evaluation_json, "metrics.json", "evaluation.json")
    resolved_optimization = _auto(optimization_json, "optimization.json")
    resolved_overlays = _auto(overlay_dir, "overlays", "masks")
    resolved_plots = _auto(plots_dir, "plots")

    found: list[str] = []
    if resolved_inspection:
        found.append(f"inspection ({resolved_inspection.name})")
    if resolved_segmentation:
        found.append(f"segmentation ({resolved_segmentation.name})")
    if resolved_evaluation:
        found.append(f"evaluation ({resolved_evaluation.name})")
    if resolved_optimization:
        found.append(f"optimization ({resolved_optimization.name})")

    if not found:
        console.print(
            "[yellow]No result files found. Run inspect/segment/evaluate first "
            "or pass explicit --inspection / --segmentation flags.[/yellow]"
        )

    with console.status("[bold green]Building report …"):
        try:
            data = load_report_data(
                inspection_json=resolved_inspection,
                segmentation_json=resolved_segmentation,
                evaluation_json=resolved_evaluation,
                optimization_json=resolved_optimization,
                project_yaml=project,
                overlay_dir=(
                    resolved_overlays if resolved_overlays and resolved_overlays.is_dir() else None
                ),
                plots_dir=resolved_plots if resolved_plots and resolved_plots.is_dir() else None,
                command="microagent report",
            )
            generate_report(data, output)
        except ImportError as exc:
            console.print(f"[bold red]Import error:[/bold red] {exc}")
            raise typer.Exit(1) from None
        except Exception as exc:
            console.print(f"[bold red]Report generation failed:[/bold red] {exc}")
            raise typer.Exit(1) from None

    console.print(
        Panel(
            f"[bold]Sections included:[/bold]  {', '.join(found) or '(provenance only)'}\n"
            f"[bold]Output:[/bold]  {output}",
            title="Report Generated",
            border_style="green",
        )
    )
    console.print(f"\n[green]✓ Report saved →[/green] {output}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@app.command()
def export(
    run: Optional[str] = typer.Option(
        None,
        "--run",
        help="8-character run ID to bundle (from experiments.jsonl)",
    ),
    fmt: str = typer.Option(
        "bundle",
        "--format",
        "-f",
        help="Export format: docker | apptainer | bundle",
    ),
    output: Path = typer.Option(
        Path("export"),
        "--output",
        "-o",
        help="Output directory for generated files",
    ),
    dockerfile_only: bool = typer.Option(
        False,
        "--dockerfile",
        help="Generate a Dockerfile from the current environment (no run ID needed)",
    ),
    experiments: Path = typer.Option(
        Path("experiments.jsonl"),
        "--experiments",
        help="Path to experiments.jsonl log",
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        help="Path to project.yaml (auto-detected if omitted)",
    ),
) -> None:
    """Export a reproducibility bundle or container definition for a run.

    Examples
    --------
    Generate a Dockerfile from the current environment:

        microagent export --dockerfile

    Create a full reproducibility bundle for run ``abc12345``:

        microagent export --run abc12345 --format bundle --output ./export/

    Generate only the Apptainer definition for an HPC cluster:

        microagent export --run abc12345 --format apptainer
    """
    from microagent.fair.container import (
        export_reproducibility_bundle,
        generate_apptainer_def,
        generate_dockerfile,
        generate_environment_lock,
    )
    from microagent.fair.provenance import collect_metadata

    output.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()

    # ── --dockerfile shortcut ─────────────────────────────────────────────────
    if dockerfile_only:
        with console.status("[bold green]Collecting environment …"):
            meta = collect_metadata(command="microagent export --dockerfile")
            dest = output / "Dockerfile"
            generate_dockerfile(meta, dest)
        console.print(
            Panel(
                f"[bold]Target:[/bold]  {dest}",
                title="Dockerfile Generated",
                border_style="green",
            )
        )
        console.print(f"\n[green]✓[/green] {dest}")
        return

    # ── run-based exports ─────────────────────────────────────────────────────
    if run is None:
        console.print(
            "[bold red]Error:[/bold red] --run <run_id> is required unless "
            "--dockerfile is specified."
        )
        raise typer.Exit(1)

    valid_formats = {"docker", "apptainer", "bundle"}
    if fmt not in valid_formats:
        console.print(
            f"[bold red]Unknown format:[/bold red] {fmt!r}. "
            f"Choose from: {', '.join(sorted(valid_formats))}"
        )
        raise typer.Exit(1)

    with console.status(f"[bold green]Loading run {run} …"):
        try:
            if fmt == "bundle":
                zip_path = export_reproducibility_bundle(
                    run_id=run,
                    output_dir=output,
                    experiments_path=experiments,
                    project_yaml=project,
                )
                console.print(
                    Panel(
                        f"[bold]Run:[/bold]  {run}\n"
                        f"[bold]Bundle:[/bold]  {zip_path}",
                        title="Reproducibility Bundle",
                        border_style="green",
                    )
                )
                console.print(f"\n[green]✓ Bundle saved →[/green] {zip_path}")

            else:
                # Load provenance from the run record
                import json as _json

                experiments_path = Path(experiments)
                if not experiments_path.exists():
                    console.print(
                        f"[bold red]Experiments log not found:[/bold red] {experiments_path}"
                    )
                    raise typer.Exit(1)

                run_record: dict | None = None
                with experiments_path.open(encoding="utf-8") as fh:
                    for _line in fh:
                        _line = _line.strip()
                        if not _line:
                            continue
                        rec = _json.loads(_line)
                        if rec.get("run_id") == run:
                            run_record = rec
                            break

                if run_record is None:
                    console.print(
                        f"[bold red]Run not found:[/bold red] '{run}' in {experiments_path}"
                    )
                    raise typer.Exit(1)

                from microagent.fair.provenance import RunMetadata

                m = run_record.get("metadata", {})
                provenance = RunMetadata(
                    microagent_version=m.get("microagent_version", "unknown"),
                    python_version=m.get("python_version", "unknown"),
                    platform=m.get("platform", "unknown"),
                    cellpose_version=m.get("cellpose_version"),
                    stardist_version=m.get("stardist_version"),
                    torch_version=m.get("torch_version", "unknown"),
                    numpy_version=m.get("numpy_version", "unknown"),
                    cuda_version=m.get("cuda_version"),
                    gpu_name=m.get("gpu_name"),
                    gpu_vram_mb=m.get("gpu_vram_mb"),
                    cpu_model=m.get("cpu_model", "unknown"),
                    ram_total_gb=m.get("ram_total_gb", 0.0),
                    data_hash=m.get("data_hash", ""),
                    parameters=m.get("parameters", {}),
                    random_seed=m.get("random_seed", 0),
                    timestamp_utc=m.get("timestamp_utc", ""),
                    wall_clock_seconds=m.get("wall_clock_seconds", 0.0),
                    git_commit=m.get("git_commit"),
                    git_dirty=m.get("git_dirty"),
                    command=m.get("command", ""),
                )

                if fmt == "docker":
                    dest = output / "Dockerfile"
                    generate_dockerfile(provenance, dest)
                    console.print(f"\n[green]✓ Dockerfile →[/green] {dest}")
                else:  # apptainer
                    dest = output / "microagent.def"
                    generate_apptainer_def(provenance, dest)
                    console.print(f"\n[green]✓ Apptainer def →[/green] {dest}")

        except (KeyError, FileNotFoundError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from None
        except Exception as exc:
            console.print(f"[bold red]Export failed:[/bold red] {exc}")
            raise typer.Exit(1) from None


@app.command()
def mcp_server(
    transport: str = typer.Option("stdio", "--transport", help="Transport type: stdio or http"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host for HTTP transport"),
    port: int = typer.Option(8080, "--port", help="Port for HTTP transport"),
) -> None:
    """Run the MicroAgent MCP server.

    Connect to any MCP-compatible client (Vibe, Claude Code, etc.).
    For Vibe, add to config.toml:

        [[mcp_servers]]
        name = "microagent"
        command = "microagent"
        args = ["mcp-server"]
    """
    try:
        from mcp.server.fastmcp import FastMCP

        _HAS_MCP = True
    except ImportError:
        console.print("[bold red]Error:[/bold red] mcp package required. Install with: pip install 'microagent[mcp]'")
        raise typer.Exit(1) from None

    from microagent.mcp_server import mcp

    if transport == "http":
        console.print(f"[bold green]Starting MCP server on http://{host}:{port}[/bold green]")
        mcp.run(transport="http", host=host, port=port)
    else:
        console.print("[bold green]Starting MCP server (stdio transport)[/bold green]")
        mcp.run(transport="stdio")
