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


@app.command()
def benchmark(
    prompt_file: Path = typer.Option(
        PROMPTS_DIR / "benchmark_set.json",
        "--prompt-file",
        "-f",
        help="JSON file with list of {prompt, reference?} objects.",
        exists=True,
    ),
) -> None:
    """Run all prompts in a JSON file and save results to data/results/."""
    prompts = json.loads(prompt_file.read_text())
    console.print(f"[bold]Running {len(prompts)} prompts…[/bold]")
    for i, entry in enumerate(prompts, 1):
        prompt = entry if isinstance(entry, str) else entry["prompt"]
        reference = None if isinstance(entry, str) else entry.get("reference")
        console.rule(f"[{i}/{len(prompts)}] {prompt[:60]}")
        _run(prompt, reference=reference, save_result=True)


if __name__ == "__main__":
    app()
