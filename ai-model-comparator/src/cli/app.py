import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer

from ..config import build_registry
from ..metrics.summary import get_full_summary
from ..runner.prompt_runner import run_comparison
from ..runner.result_collector import collect, save
from .display import console, print_results

app = typer.Typer(help="AI Model Comparator — benchmark LLM responses across cost, speed, and quality.")

PROMPTS_DIR = Path(__file__).parents[2] / "data" / "prompts"


def _run(prompt: str, reference: str | None = None, save_result: bool = True) -> None:
    registry = build_registry()
    run_results = asyncio.run(run_comparison(prompt, registry, timeout=60.0))
    result = collect(prompt, run_results, reference_answer=reference)
    if save_result:
        path = save(result)
        console.print(f"[dim]saved → {path}[/dim]")
    df = get_full_summary(result, reference=reference)
    print_results(result, df)


@app.command()
def compare(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to send to all models."),
    reference: Optional[str] = typer.Option(None, "--reference", "-r", help="Optional ground-truth for ROUGE scoring."),
) -> None:
    """Send a single prompt to all models and display ranked results."""
    _run(prompt, reference=reference)


def _load_prompts(path: Path) -> list[dict]:
    """Load prompts from a .json or .txt file.

    JSON: list of strings or {prompt, reference?} objects.
    TXT:  one prompt per non-empty line.
    """
    if path.suffix == ".txt":
        entries = [{"prompt": line} for line in path.read_text().splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text())
        entries = [{"prompt": e} if isinstance(e, str) else e for e in raw]
    return entries


@app.command()
def benchmark(
    prompt_file: Path = typer.Option(
        PROMPTS_DIR / "benchmark_set.json",
        "--prompt-file",
        "-f",
        help="Prompt file to run (.json list or .txt one-per-line).",
    ),
) -> None:
    """Run all prompts in a file and save each result to data/results/."""
    if not prompt_file.exists():
        console.print(f"[red]File not found:[/red] {prompt_file}")
        raise typer.Exit(1)
    entries = _load_prompts(prompt_file)
    console.print(f"[bold]Running {len(entries)} prompts from {prompt_file.name}…[/bold]")
    for i, entry in enumerate(entries, 1):
        prompt = entry["prompt"]
        reference = entry.get("reference")
        console.rule(f"[{i}/{len(entries)}] {prompt[:60]}")
        _run(prompt, reference=reference, save_result=True)


if __name__ == "__main__":
    app()
