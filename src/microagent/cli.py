import typer

app = typer.Typer(
    name="microagent",
    help="Agentic microscopy image analysis tool.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
