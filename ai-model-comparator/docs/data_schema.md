# Data Schema

## Overview

All data flows through three core structures:

```
PromptRequest
    └── run_comparison()
            ├── ModelResponse   (one per model, raw API output)
            └── ComparisonResult (one per run, aggregated + scored)
```

---

## 1. `PromptRequest`

Input to the comparison runner.

```python
@dataclass
class PromptRequest:
    prompt: str                        # The user-supplied input text
    max_tokens: int = 1024             # Max tokens to request from each model
    reference_answer: str | None = None  # Optional ground-truth for ROUGE scoring
    run_id: str = field(default_factory=lambda: uuid4().hex[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | `str` | Yes | Text sent to all models |
| `max_tokens` | `int` | No (default: 1024) | Max output tokens per model |
| `reference_answer` | `str \| None` | No | Ground-truth for ROUGE-L scoring |
| `run_id` | `str` | Auto | Short unique ID for this run (8-char hex) |
| `timestamp` | `str` | Auto | ISO 8601 UTC timestamp |

---

## 2. `ModelResponse`

Raw output from a single model call. One instance per model per run.

```python
@dataclass
class ModelResponse:
    model_id: str          # e.g. "gpt-4o-mini"
    provider: str          # e.g. "openai" | "anthropic" | "ollama"
    text: str              # The model's response text
    input_tokens: int      # Tokens consumed from the prompt
    output_tokens: int     # Tokens generated in the response
    latency_ms: int        # Wall-clock time in milliseconds
    error: str | None      # Error message if call failed, else None
```

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | Model identifier string |
| `provider` | `str` | API provider name |
| `text` | `str` | Full response text (empty string on error) |
| `input_tokens` | `int` | From API usage field or `tiktoken` estimate |
| `output_tokens` | `int` | From API usage field or `tiktoken` estimate |
| `latency_ms` | `int` | Elapsed time from request send to response complete |
| `error` | `str \| None` | `None` on success; exception message on failure |

---

## 3. `MetricsResult`

Computed metrics for a single `ModelResponse`. One instance per model per run.

```python
@dataclass
class MetricsResult:
    model_id: str
    cost_usd: float        # Calculated from token counts × pricing table
    rouge_l: float | None  # ROUGE-L F1 score [0.0, 1.0]; None if no reference
    judge_score: float | None  # LLM-as-judge average [1.0, 5.0]; None if skipped
    composite_score: float | None  # Weighted blend [0.0, 1.0]; None if insufficient data
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `cost_usd` | `float` | ≥ 0.0 | Estimated USD cost of this call |
| `rouge_l` | `float \| None` | 0.0–1.0 | ROUGE-L F1 vs reference or consensus |
| `judge_score` | `float \| None` | 1.0–5.0 | LLM-as-judge average rating |
| `composite_score` | `float \| None` | 0.0–1.0 | Weighted composite (Quality 50%, Latency 30%, Cost 20%) |

---

## 4. `ComparisonResult`

Top-level aggregation for a single prompt run across all models.

```python
@dataclass
class ComparisonResult:
    run_id: str
    timestamp: str
    prompt: str
    reference_answer: str | None
    responses: list[ModelResponse]   # One per model
    metrics: list[MetricsResult]     # One per model
    rankings: dict[str, list[str]]   # {"latency": [...], "cost": [...], "quality": [...], "composite": [...]}
```

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Matches `PromptRequest.run_id` |
| `timestamp` | `str` | ISO 8601 UTC |
| `prompt` | `str` | The original prompt text |
| `reference_answer` | `str \| None` | Ground-truth if provided |
| `responses` | `list[ModelResponse]` | One entry per model (including failed ones) |
| `metrics` | `list[MetricsResult]` | Corresponding metrics per model |
| `rankings` | `dict[str, list[str]]` | Model IDs sorted best-to-worst per metric |

---

## 5. Serialized JSON Format

`ComparisonResult` is persisted as JSON to `data/results/{run_id}_{timestamp}.json`.

```json
{
  "run_id": "a3f9c2d1",
  "timestamp": "2025-03-11T10:22:00Z",
  "prompt": "Explain recursion in one paragraph.",
  "reference_answer": null,
  "responses": [
    {
      "model_id": "gpt-4o-mini",
      "provider": "openai",
      "text": "Recursion is a technique...",
      "input_tokens": 12,
      "output_tokens": 98,
      "latency_ms": 821,
      "error": null
    },
    {
      "model_id": "claude-3-5-haiku-20241022",
      "provider": "anthropic",
      "text": "Recursion refers to...",
      "input_tokens": 12,
      "output_tokens": 104,
      "latency_ms": 1103,
      "error": null
    },
    {
      "model_id": "llama3",
      "provider": "ollama",
      "text": "In programming, recursion means...",
      "input_tokens": 11,
      "output_tokens": 89,
      "latency_ms": 3240,
      "error": null
    }
  ],
  "metrics": [
    {
      "model_id": "gpt-4o-mini",
      "cost_usd": 0.00006090,
      "rouge_l": null,
      "judge_score": 4.3,
      "composite_score": 0.81
    },
    {
      "model_id": "claude-3-5-haiku-20241022",
      "cost_usd": 0.00042560,
      "rouge_l": null,
      "judge_score": 4.5,
      "composite_score": 0.76
    },
    {
      "model_id": "llama3",
      "cost_usd": 0.00000000,
      "rouge_l": null,
      "judge_score": 3.8,
      "composite_score": 0.62
    }
  ],
  "rankings": {
    "latency":   ["gpt-4o-mini", "claude-3-5-haiku-20241022", "llama3"],
    "cost":      ["llama3", "gpt-4o-mini", "claude-3-5-haiku-20241022"],
    "quality":   ["claude-3-5-haiku-20241022", "gpt-4o-mini", "llama3"],
    "composite": ["gpt-4o-mini", "claude-3-5-haiku-20241022", "llama3"]
  }
}
```

---

## 6. File Naming Convention

```
data/results/{run_id}_{YYYYMMDD_HHMMSS}.json

# Example:
data/results/a3f9c2d1_20250311_102200.json
```

Benchmark runs (multiple prompts) produce one file per prompt, all prefixed with the same benchmark session ID:

```
data/results/bench_{session_id}_{run_id}_{YYYYMMDD_HHMMSS}.json
```

---

## 7. Implementation Location

| Structure | Defined in |
|-----------|-----------|
| `PromptRequest` | `src/runner/prompt_runner.py` |
| `ModelResponse` | `src/models/base.py` |
| `MetricsResult` | `src/metrics/__init__.py` |
| `ComparisonResult` | `src/runner/result_collector.py` |
