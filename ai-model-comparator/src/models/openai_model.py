from time import time
import time
from openai import AsyncOpenAI

from .base import BaseModel, ModelResponse

MODEL_ID = "gpt-4o-mini"

class OpenAIModel(BaseModel):
    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str) -> ModelResponse:
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        return ModelResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency,
            model_id=MODEL_ID
        )
        


