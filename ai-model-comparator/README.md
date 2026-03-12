# AI Model Comparator

A Python tool that compares responses from multiple AI models side-by-side, measuring **cost**, **speed**, and **quality** on any prompt.

## Models Compared

| Model | Provider | Type |
|-------|----------|------|
| `gpt-4o-mini` | OpenAI | Proprietary API |
| `claude-3-5-haiku-20241022` | Anthropic | Proprietary API |
| `llama3` | Meta via Ollama | Open-source, local |

## Features

- Send any prompt to all models concurrently
- Measure latency, cost, and quality (ROUGE-L + LLM-as-judge)
- Rich CLI output with color-coded rankings
- Benchmark mode for batch evaluation
- Jupyter notebook for detailed analysis

## Setup

```bash
# 1. Clone and activate venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. Configure secrets
cp .env.example .env  # then fill in your API keys

# 3. Install Ollama and pull llama3 (local open-source model)
brew install ollama   # or https://ollama.com
ollama pull llama3
ollama serve          # runs on http://localhost:11434
```

## License

MIT
