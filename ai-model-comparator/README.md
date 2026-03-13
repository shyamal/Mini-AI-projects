# AI Model Comparator

A Python CLI tool that benchmarks AI model responses across **cost**, **speed**, and **quality** — concurrently, on any prompt.

```
┌──────────────────────────── Model Responses ──────────────────────────────────┐
│ Provider  │ Model ID                  │ Response                              │
│ openai    │ gpt-4o-mini               │ Recursion is when a function calls... │
│ anthropic │ claude-haiku-4-5-20251001 │ Recursion is a programming technique  │
│ ollama    │ llama3                    │ Recursion is a concept where a func.. │
├──────────────────────────── Metrics Summary ──────────────────────────────────┤
│ Provider  │   Latency │  Cost (USD) │ ROUGE-L │ Composite │                   │
│ openai    │  1,914 ms │  $0.000007  │  1.0000 │    0.8000 │  ← best overall   │
│ anthropic │  2,103 ms │  $0.000032  │  0.8412 │    0.6200 │                   │
│ ollama    │  9,903 ms │  $0.000000  │  0.6731 │    0.4728 │                   │
└───────────────────────────────────────────────────────────────────────────────┘
  Rankings  Speed: openai › anthropic › ollama
            Cost:  ollama › openai › anthropic
            Overall: openai › anthropic › ollama
```

## Models

| Model | Provider | Input / Output (per 1M tokens) |
|-------|----------|--------------------------------|
| `gpt-4o-mini` | OpenAI API | $0.15 / $0.60 |
| `claude-haiku-4-5-20251001` | Anthropic API | $0.80 / $4.00 |
| `llama3` | Ollama (local) | Free |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed locally
- OpenAI API key
- Anthropic API key

## Setup

```bash
# 1. Clone and activate the virtual environment
git clone <repo-url> && cd ai-model-comparator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and ANTHROPIC_API_KEY

# 3. Start the local model
ollama pull llama3
ollama serve          # listens on http://localhost:11434
```

## Usage

### Compare a single prompt

```bash
python -m src.cli.app compare --prompt "Explain recursion in one sentence."

# With a reference answer for ROUGE-L scoring
python -m src.cli.app compare \
  --prompt "What year did the Berlin Wall fall?" \
  --reference "The Berlin Wall fell in 1989."
```

### Run the benchmark set

```bash
# Runs all 15 prompts in data/prompts/benchmark_set.json
python -m src.cli.app benchmark

# Custom prompt file — JSON or plain text
python -m src.cli.app benchmark --prompt-file my_prompts.txt
```

### Prompt file formats

**JSON** — strings or objects with optional reference answers:
```json
[
  "What is 2+2?",
  {"prompt": "What year did WWII end?", "reference": "1945"}
]
```

**TXT** — one prompt per line:
```
What is 2+2?
Explain recursion in one sentence.
```

## Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| Latency | Wall-clock ms from request to full response | Lower |
| Cost | Token counts × published pricing (USD) | Lower |
| ROUGE-L | LCS F1 score vs reference or consensus response | Higher |
| Composite | Quality 50% + Latency 30% + Cost 20% (normalised) | Higher |

## Project Structure

```
src/
├── models/          # BaseModel + OpenAI, Anthropic, Ollama clients
├── runner/          # Concurrent runner + result collector
├── metrics/         # Latency, cost, quality, composite summary
└── cli/             # Typer commands + Rich display

data/
├── prompts/         # benchmark_set.json (tracked in git)
└── results/         # Per-run JSON output (gitignored)

notebooks/
└── evaluation.ipynb # Latency / cost / quality charts + ranking table
```

## Analysis Notebook

After running benchmarks, open the notebook to explore the saved results:

```bash
jupyter notebook notebooks/evaluation.ipynb
```

Covers: data loading · latency chart · cost chart · ROUGE-L chart · overall ranking table · conclusions.

## License

MIT
