import asyncio
from dataclasses import dataclass
from ..models.base import BaseModel, ModelResponse

DEFAULT_TIMEOUT = 30.0  # seconds


@dataclass
class RunResult:
    key: str
    response: ModelResponse | None
    error: str | None


async def _run_one(key: str, model: BaseModel, prompt: str, timeout: float) -> RunResult:
    try:
        response = await asyncio.wait_for(model.generate(prompt), timeout=timeout)
        return RunResult(key=key, response=response, error=None)
    except asyncio.TimeoutError:
        return RunResult(key=key, response=None, error=f"timed out after {timeout}s")
    except Exception as exc:
        return RunResult(key=key, response=None, error=str(exc))


async def run_comparison(
    prompt: str,
    registry: dict[str, BaseModel],
    timeout: float = DEFAULT_TIMEOUT,
) -> list[RunResult]:
    tasks = [_run_one(key, model, prompt, timeout) for key, model in registry.items()]
    return await asyncio.gather(*tasks)
