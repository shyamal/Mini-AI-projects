from models.openai_model import MODEL_ID
import time
import anthropic
from .base import BaseModel, ModelResponse

MODEL_ID = "claude-haiku-4-5-20251001"

class AnthropicModel(BaseModel):
    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str) -> ModelResponse:
        start = time.monotonic()
        response = await self._client.messages.create(
            model=MODEL_ID,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.monotonic() - start) * 1000

        return ModelResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency,
            model_id=MODEL_ID
        