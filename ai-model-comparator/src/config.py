import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .models.anthropic_model import AnthropicModel
from .models.base import BaseModel
from .models.openai_model import OpenAIModel
from .models.opensource_model import OllamaModel

load_dotenv()


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float   # USD per 1K input tokens
    output_per_1k: float  # USD per 1K output tokens


PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini":              ModelPricing(input_per_1k=0.000150, output_per_1k=0.000600),
    "claude-haiku-4-5-20251001": ModelPricing(input_per_1k=0.000800, output_per_1k=0.004000),
    "llama3":                   ModelPricing(input_per_1k=0.0,       output_per_1k=0.0),
}


def build_registry() -> dict[str, BaseModel]:
    return {
        "openai":    OpenAIModel(api_key=os.environ["OPENAI_API_KEY"]),
        "anthropic": AnthropicModel(api_key=os.environ["ANTHROPIC_API_KEY"]),
        "ollama":    OllamaModel(),
    }
