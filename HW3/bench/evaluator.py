from typing import Optional
from bench.openrouter import OpenRouterClient
from bench.prompts import VERIFY_SYSTEM_PROMPT, VERIFY_USER_TEMPLATE
from bench.config import VERIFIER_MODEL, MAX_TOKENS_VERIFY


_EXPECTED_CLASSIFICATIONS = frozenset(
    {"CORRECT", "PARTIALLY_CORRECT", "INCORRECT"}
)


class LLMEvaluator:
    def __init__(
        self, client: OpenRouterClient, model: Optional[str] = None
    ):
        self._client = client
        self._model = model or VERIFIER_MODEL

    def evaluate(
        self,
        question: str,
        ground_truth: str,
        observations: str,
        model_answer: str,
        csv_row_indices: str = "",
    ) -> dict:
        user_message = VERIFY_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            observations=observations,
            csv_row_indices=csv_row_indices,
            model_answer=model_answer,
        )

        result = self._client.chat(
            model=self._model,
            system_prompt=VERIFY_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=MAX_TOKENS_VERIFY,
        )

        classification = "INCORRECT"
        explanation = ""

        for raw_line in result["content"].split("\n"):
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("CLASSIFICATION:"):
                label = line.split(":", 1)[1].strip().upper()
                classification = _normalize_classification(label)
            if upper.startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()

        return {
            "classification": classification,
            "explanation": explanation,
            "raw_output": result["content"],
            "tokens": result["tokens_total"],
            "cost": result["cost"],
        }


def _normalize_classification(raw: str) -> str:
    upper = raw.upper()
    if "INCORRECT" in upper:
        return "INCORRECT"
    if "PARTIALLY" in upper:
        return "PARTIALLY_CORRECT"
    if "CORRECT" in upper:
        return "CORRECT"
    return "INCORRECT"
