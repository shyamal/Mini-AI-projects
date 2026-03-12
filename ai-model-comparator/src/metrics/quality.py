import json

from openai import AsyncOpenAI
from rouge_score import rouge_scorer

from ..runner.result_collector import NormalizedResponse

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _rouge_l(reference: str, hypothesis: str) -> float:
    return _scorer.score(reference, hypothesis)["rougeL"].fmeasure


def _consensus_reference(responses: list[NormalizedResponse]) -> str:
    """Pick the response with highest average ROUGE-L similarity to all others."""
    texts = [r.text for r in responses]
    if len(texts) == 1:
        return texts[0]
    best, best_score = texts[0], -1.0
    for candidate in texts:
        avg = sum(_rouge_l(candidate, other) for other in texts if other != candidate) / (len(texts) - 1)
        if avg > best_score:
            best, best_score = candidate, avg
    return best


def get_quality_summary(
    responses: list[NormalizedResponse],
    reference: str | None = None,
) -> dict:
    successful = [r for r in responses if r.error is None and r.text]
    if not successful:
        return {"ranked": [], "by_provider": {}}

    ref = reference if reference is not None else _consensus_reference(successful)
    scores = {r.provider: round(_rouge_l(ref, r.text), 4) for r in successful}
    ranked = sorted(scores, key=lambda p: scores[p], reverse=True)
    return {"ranked": ranked, "by_provider": scores, "reference_used": ref}


_JUDGE_PROMPT = """\
You are an impartial AI evaluator. Rate the following response on a scale of 1–5 for:
1. Accuracy – Is the information correct?
2. Clarity – Is the response easy to understand?
3. Completeness – Does it fully address the question?

Question: {prompt}
Response: {response}

Return JSON only: {{"accuracy": int, "clarity": int, "completeness": int}}"""


async def get_judge_scores(
    prompt: str,
    responses: list[NormalizedResponse],
    api_key: str,
) -> dict:
    client = AsyncOpenAI(api_key=api_key)
    scores: dict[str, float | None] = {}

    for r in responses:
        if r.error or not r.text:
            scores[r.provider] = None
            continue
        try:
            reply = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": _JUDGE_PROMPT.format(prompt=prompt, response=r.text)}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(reply.choices[0].message.content)
            scores[r.provider] = round(sum(data[k] for k in ("accuracy", "clarity", "completeness")) / 3, 2)
        except Exception as exc:
            scores[r.provider] = None

    ranked = sorted([p for p, s in scores.items() if s is not None], key=lambda p: scores[p], reverse=True)
    return {"ranked": ranked, "by_provider": scores}
