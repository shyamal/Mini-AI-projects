import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .prompt_runner import RunResult

RESULTS_DIR = Path(__file__).parents[2] / "data" / "results"


@dataclass
class NormalizedResponse:
    model_id: str
    provider: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None


@dataclass
class ComparisonResult:
    run_id: str
    timestamp: str
    prompt: str
    reference_answer: str | None
    responses: list[NormalizedResponse]
    metrics: list[dict] = field(default_factory=list)
    rankings: dict[str, list[str]] = field(default_factory=dict)


def collect(
    prompt: str,
    run_results: list[RunResult],
    reference_answer: str | None = None,
    run_id: str | None = None,
) -> ComparisonResult:
    if run_id is None:
        run_id = uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    responses = []
    for r in run_results:
        if r.response is not None:
            responses.append(NormalizedResponse(
                model_id=r.response.model_id,
                provider=r.key,
                text=r.response.text,
                input_tokens=r.response.input_tokens,
                output_tokens=r.response.output_tokens,
                latency_ms=r.response.latency_ms,
                error=None,
            ))
        else:
            responses.append(NormalizedResponse(
                model_id=r.key,
                provider=r.key,
                text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                error=r.error,
            ))

    return ComparisonResult(
        run_id=run_id,
        timestamp=timestamp,
        prompt=prompt,
        reference_answer=reference_answer,
        responses=responses,
    )


def save(result: ComparisonResult, directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = directory / f"{result.run_id}_{ts}.json"
    path.write_text(json.dumps(asdict(result), indent=2))
    return path
