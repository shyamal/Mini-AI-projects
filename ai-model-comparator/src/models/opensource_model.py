import time

import httpx
import tiktoken

from .base import BaseModel, ModelResponse

MODEL_ID = "llama3"
OLLAMA_URL = "http://localhost:11434/api/chat"

_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


class OllamaModel(BaseModel):
    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self._url = base_url

    async def generate(self, prompt: str) -> ModelResponse:
        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self._url, json=payload)
            response.raise_for_status()
        latency_ms = (time.monotonic() - start) * 1000

        data = response.json()
        text = data["message"]["content"]
        return ModelResponse(
            text=text,
            input_tokens=_count_tokens(prompt),
            output_tokens=_count_tokens(text),
            latency_ms=latency_ms,
            model_id=MODEL_ID,
        )
