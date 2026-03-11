# Model Matrix

## Models Under Comparison

| # | Provider | Model ID | Type | Context Window |
|---|----------|----------|------|----------------|
| 1 | OpenAI | `gpt-4o-mini` | Proprietary API | 128,000 tokens |
| 2 | Anthropic | `claude-3-5-haiku-20241022` | Proprietary API | 200,000 tokens |
| 3 | Meta (via Ollama) | `llama3` | Open-source, local | 8,192 tokens |

> **Fallback for model 3:** `mistralai/Mistral-7B-Instruct-v0.3` via HuggingFace Inference API (32,768 token context) if Ollama is unavailable.

---

## Pricing Table

### OpenAI — `gpt-4o-mini`

| Token Type | Price per 1M tokens | Price per 1K tokens |
|------------|--------------------|--------------------|
| Input | $0.150 | $0.000150 |
| Output | $0.600 | $0.000600 |

### Anthropic — `claude-3-5-haiku-20241022`

| Token Type | Price per 1M tokens | Price per 1K tokens |
|------------|--------------------|--------------------|
| Input | $0.800 | $0.000800 |
| Output | $4.000 | $0.004000 |

### Meta — `llama3` via Ollama (local)

| Token Type | Price |
|------------|-------|
| Input | $0.00 (local inference) |
| Output | $0.00 (local inference) |

> Note: Local cost is effectively $0 but compute/electricity cost applies. If using HuggingFace Inference API free tier, rate limits apply (1,000 req/day).

---

## Capability Summary

| Capability | gpt-4o-mini | claude-3-5-haiku | llama3 (local) |
|------------|-------------|------------------|----------------|
| Reasoning | Strong | Strong | Moderate |
| Code generation | Strong | Strong | Moderate |
| Instruction following | Excellent | Excellent | Good |
| Long context | 128K | 200K | 8K |
| Async API | Yes | Yes | Yes (HTTP) |
| Token usage in response | Yes | Yes | Estimated |
| Cost | Low | Medium | Free |
| Latency (expected) | Low | Low | Varies (hardware) |

---

## API Access Requirements

| Model | Credential | Where to get |
|-------|-----------|--------------|
| `gpt-4o-mini` | `OPENAI_API_KEY` | platform.openai.com |
| `claude-3-5-haiku-20241022` | `ANTHROPIC_API_KEY` | console.anthropic.com |
| `llama3` (Ollama) | None | Install Ollama locally: `brew install ollama` |
| Mistral (HuggingFace fallback) | `HUGGINGFACE_API_KEY` | huggingface.co/settings/tokens |

---

## Notes

- Pricing is accurate as of Q1 2025. Verify current rates before running cost benchmarks.
- Token counts for Ollama responses must be estimated using `tiktoken` (cl100k_base encoding) as the Ollama API does not always return exact counts.
- Context window limits should be enforced in `src/config.py` to prevent oversized prompt errors.
