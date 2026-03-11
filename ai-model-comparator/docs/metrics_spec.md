# Metrics Specification

## Overview

Each model response is evaluated on three dimensions: **Latency**, **Cost**, and **Quality**.
All metrics are stored per response in the `ComparisonResult` data structure.

---

## 1. Latency

**Definition:** Wall-clock elapsed time from the moment the API request is sent to the moment the full response is received.

**Unit:** Milliseconds (`int`)

**Implementation:**

```python
import time

start = time.perf_counter()
response = await model.generate(prompt)
latency_ms = int((time.perf_counter() - start) * 1000)
```

**Notes:**
- Measures total response time, not time-to-first-token (TTFT), since not all APIs support streaming in v1.
- Streaming support and TTFT measurement can be added as a future enhancement.
- Network conditions affect latency; local models (Ollama) avoid network overhead but are bound by hardware.

**Ranking:** Lower is better.

---

## 2. Cost

**Definition:** Estimated USD cost of a single API call, calculated from token counts and published pricing.

**Unit:** USD (`float`, rounded to 8 decimal places)

**Formula:**

```
cost = (input_tokens  / 1_000_000 * input_price_per_million)
     + (output_tokens / 1_000_000 * output_price_per_million)
```

**Pricing constants** (defined in `src/config.py`):

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|--------------|
| `gpt-4o-mini` | 0.150 | 0.600 |
| `claude-3-5-haiku-20241022` | 0.800 | 4.000 |
| `llama3` (Ollama local) | 0.000 | 0.000 |

**Token source per model:**

| Model | Token count source |
|-------|--------------------|
| `gpt-4o-mini` | `response.usage.prompt_tokens` / `completion_tokens` |
| `claude-3-5-haiku-20241022` | `response.usage.input_tokens` / `output_tokens` |
| `llama3` (Ollama) | Estimated via `tiktoken` (cl100k_base) |

**Ranking:** Lower is better.

---

## 3. Quality

**Definition:** A measure of how accurate, relevant, and useful the model's response is relative to a reference or peer responses.

**Unit:** Float in range `[0.0, 1.0]` (ROUGE-L) or `[1, 5]` (LLM-as-judge)

### 3a. ROUGE-L Score (automatic, no reference required for relative comparison)

- Uses longest common subsequence (LCS) overlap between two texts.
- In **reference mode**: compare each model response against a known ground-truth answer.
- In **cross-model mode** (no reference): use the majority/consensus response as a soft reference.
- Library: `rouge-score` (`pip install rouge-score`)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
score = scorer.score(reference, hypothesis)
rouge_l = score["rougeL"].fmeasure  # float 0.0–1.0
```

**Ranking:** Higher is better.

### 3b. LLM-as-Judge Score (optional, requires additional API call)

- Send all model responses for a given prompt to `gpt-4o-mini` with a structured evaluation prompt.
- Request a score from 1–5 on: accuracy, clarity, completeness.
- Average the three sub-scores into a single `judge_score`.

**Judge prompt template:**

```
You are an impartial AI evaluator. Rate the following response on a scale of 1–5 for:
1. Accuracy – Is the information correct?
2. Clarity – Is the response easy to understand?
3. Completeness – Does it fully address the question?

Question: {prompt}
Response: {response}

Return JSON: {"accuracy": int, "clarity": int, "completeness": int}
```

**Ranking:** Higher is better.

---

## Composite Ranking

For each prompt run, models are ranked across all three dimensions and assigned a weighted composite score:

| Metric | Weight |
|--------|--------|
| Quality (ROUGE-L or judge) | 50% |
| Latency | 30% |
| Cost | 20% |

> Weights are configurable in `src/config.py` under `METRIC_WEIGHTS`.

**Normalization before weighting:**
- Latency and cost are normalized to `[0, 1]` (inverted — lower raw = higher score).
- Quality is already in `[0, 1]` range.

---

## Output Schema

Each metric result is stored alongside the response:

```json
{
  "model_id": "gpt-4o-mini",
  "latency_ms": 843,
  "input_tokens": 21,
  "output_tokens": 134,
  "cost_usd": 0.00008,
  "rouge_l": 0.61,
  "judge_score": 4.0,
  "composite_score": 0.74
}
```

---

## Future Enhancements

- Time-to-first-token (TTFT) via streaming APIs
- BERTScore for semantic similarity
- Human evaluation integration
