from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latancy_ms: float
    model_id: str

    class BaseModel(ABC):
        @abstractmethod
        async def generate(self, prompt: str) -> ModelResponse:
            ...
