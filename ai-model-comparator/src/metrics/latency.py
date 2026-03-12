from ..runner.result_collector import NormalizedResponse


def get_latency_summary(responses: list[NormalizedResponse]) -> dict:
    successful = [r for r in responses if r.error is None]
    ranked = sorted(successful, key=lambda r: r.latency_ms)
    return {
        "ranked": [r.provider for r in ranked],
        "by_provider": {r.provider: round(r.latency_ms, 1) for r in responses},
    }
