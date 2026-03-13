import math

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..runner.result_collector import ComparisonResult, NormalizedResponse

console = Console()

_MEDAL = {0: "[bold yellow]1st[/]", 1: "[bold white]2nd[/]", 2: "[bold yellow3]3rd[/]"}


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt_latency(ms) -> Text:
    if _is_missing(ms):
        return Text("—", style="dim")
    return Text(f"{ms:,.0f} ms")


def _fmt_cost(usd) -> Text:
    if _is_missing(usd):
        return Text("—", style="dim")
    return Text(f"${usd:.6f}")


def _fmt_score(val) -> Text:
    if _is_missing(val):
        return Text("—", style="dim")
    return Text(f"{val:.4f}")


def _winner_style(provider: str, ranked: list[str]) -> str:
    if not ranked or ranked[0] != provider:
        return ""
    return "bold green"


def print_results(result: ComparisonResult, df: pd.DataFrame) -> None:
    # ── Response table ────────────────────────────────────────────────────────
    resp_table = Table(title="Model Responses", expand=True, show_lines=True)
    resp_table.add_column("Provider", style="bold cyan", no_wrap=True)
    resp_table.add_column("Model ID", style="dim")
    resp_table.add_column("Response")

    resp_by_provider = {r.provider: r for r in result.responses}
    for provider in df.index:
        r = resp_by_provider[provider]
        text = r.text[:300] + ("…" if len(r.text) > 300 else "") if r.text else f"[red]{r.error}[/red]"
        resp_table.add_row(provider, r.model_id, text)

    console.print(resp_table)

    # ── Metrics table ─────────────────────────────────────────────────────────
    lat_ranked = df["latency_ms"].dropna().sort_values().index.tolist()
    cost_ranked = df["cost_usd"].dropna().sort_values().index.tolist()
    qual_ranked = df["rouge_l"].dropna().sort_values(ascending=False).index.tolist()
    comp_ranked = df["composite_score"].dropna().sort_values(ascending=False).index.tolist()

    metrics_table = Table(title="Metrics Summary", expand=True)
    metrics_table.add_column("Provider", style="bold cyan", no_wrap=True)
    metrics_table.add_column("Latency", justify="right")
    metrics_table.add_column("Cost (USD)", justify="right")
    metrics_table.add_column("ROUGE-L", justify="right")
    metrics_table.add_column("Judge", justify="right")
    metrics_table.add_column("Composite", justify="right")

    for provider in df.index:
        row = df.loc[provider]
        def _styled(t: Text, style: str) -> Text:
            if style:
                t.stylize(style)
            return t

        metrics_table.add_row(
            provider,
            _styled(_fmt_latency(row["latency_ms"]), _winner_style(provider, lat_ranked)),
            _styled(_fmt_cost(row["cost_usd"]), _winner_style(provider, cost_ranked)),
            _styled(_fmt_score(row["rouge_l"]), _winner_style(provider, qual_ranked)),
            _fmt_score(row["judge_score"]),
            _styled(_fmt_score(row["composite_score"]), _winner_style(provider, comp_ranked)),
        )

    console.print(metrics_table)

    # ── Rankings panel ────────────────────────────────────────────────────────
    lines = []
    for label, ranked in [("Speed", lat_ranked), ("Cost", cost_ranked), ("Quality", qual_ranked), ("Overall", comp_ranked)]:
        medals = "  ".join(f"{_MEDAL.get(i, str(i+1))} {p}" for i, p in enumerate(ranked))
        lines.append(f"[bold]{label:<10}[/]  {medals}")

    console.print(Panel("\n".join(lines), title="Rankings", border_style="bright_blue"))
