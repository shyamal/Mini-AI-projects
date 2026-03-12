import pandas as pd

from ..runner.result_collector import ComparisonResult, NormalizedResponse
from .cost import calc_cost, get_cost_summary
from .latency import get_latency_summary
from .quality import get_quality_summary

# Weights per metrics_spec: quality 50%, latency 30%, cost 20%
METRIC_WEIGHTS = {"quality": 0.50, "latency": 0.30, "cost": 0.20}


def _normalize_invert(values: dict[str, float]) -> dict[str, float]:
    """Normalize to [0,1] and invert so lower raw value = higher score."""
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 1.0 for k in values}
    return {k: 1.0 - (v - lo) / (hi - lo) for k, v in values.items()}


def get_full_summary(
    result: ComparisonResult,
    reference: str | None = None,
    judge_scores: dict | None = None,
) -> pd.DataFrame:
    responses = result.responses
    providers = [r.provider for r in responses]

    latency_raw = {r.provider: r.latency_ms for r in responses if r.error is None}
    cost_raw = {r.provider: calc_cost(r) for r in responses if r.error is None}
    rouge = get_quality_summary(responses, reference)
    quality_raw = rouge["by_provider"]

    # Use judge scores if provided, otherwise fall back to ROUGE-L
    if judge_scores:
        quality_raw = {p: s for p, s in judge_scores["by_provider"].items() if s is not None}
        # Normalize judge scores from [1,5] to [0,1]
        quality_raw = {p: (s - 1) / 4 for p, s in quality_raw.items()}

    lat_norm = _normalize_invert(latency_raw) if latency_raw else {}
    cost_norm = _normalize_invert(cost_raw) if cost_raw else {}

    rows = []
    for r in responses:
        p = r.provider
        lat_score = lat_norm.get(p)
        cost_score = cost_norm.get(p)
        qual_score = quality_raw.get(p)

        if lat_score is not None and cost_score is not None and qual_score is not None:
            composite = round(
                METRIC_WEIGHTS["quality"] * qual_score
                + METRIC_WEIGHTS["latency"] * lat_score
                + METRIC_WEIGHTS["cost"] * cost_score,
                4,
            )
        else:
            composite = None

        rows.append({
            "provider": p,
            "model_id": r.model_id,
            "latency_ms": round(r.latency_ms, 1) if r.error is None else None,
            "cost_usd": calc_cost(r) if r.error is None else None,
            "rouge_l": quality_raw.get(p) if not judge_scores else rouge["by_provider"].get(p),
            "judge_score": judge_scores["by_provider"].get(p) if judge_scores else None,
            "composite_score": composite,
            "error": r.error,
        })

    df = pd.DataFrame(rows).set_index("provider")
    return df.sort_values("composite_score", ascending=False, na_position="last")
