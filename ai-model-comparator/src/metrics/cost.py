from ..config import PRICING
from ..runner.result_collector import NormalizedResponse


def calc_cost(response: NormalizedResponse) -> float:
    pricing = PRICING.get(response.model_id)
    if pricing is None or response.error is not None:
        return 0.0
    cost = (response.input_tokens / 1000 * pricing.input_per_1k) + \
           (response.output_tokens / 1000 * pricing.output_per_1k)
    return round(cost, 8)


def get_cost_summary(responses: list[NormalizedResponse]) -> dict:
    costs = {r.provider: calc_cost(r) for r in responses}
    ranked = sorted(
        [r.provider for r in responses if r.error is None],
        key=lambda p: costs[p],
    )
    return {"ranked": ranked, "by_provider": costs}
