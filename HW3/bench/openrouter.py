import time
from typing import Optional
from openai import OpenAI
from bench.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    RATE_LIMIT_DELAY,
    MAX_RETRIES,
    RETRY_BACKOFF,
    REQUEST_TIMEOUT,
    MAX_TOKENS_ANSWER,
    TEMPERATURE,
    MODEL_PRICING,
)


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._client = OpenAI(
            base_url=base_url or OPENROUTER_BASE_URL,
            api_key=api_key or OPENROUTER_API_KEY,
            timeout=REQUEST_TIMEOUT,
        )
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = MAX_TOKENS_ANSWER,
        temperature: float = TEMPERATURE,
    ) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return self._parse_response(model, response)

            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"  [retry {attempt + 1}/{MAX_RETRIES}] "
                        f"{exc}"
                    )
                    time.sleep(RETRY_BACKOFF)

        raise last_error  # type: ignore[misc]

    def _parse_response(self, model: str, response) -> dict:
        choice = response.choices[0]
        content = choice.message.content or ""

        tokens_prompt = 0
        tokens_completion = 0
        tokens_total = 0

        if response.usage:
            tokens_prompt = response.usage.prompt_tokens or 0
            tokens_completion = response.usage.completion_tokens or 0
            tokens_total = response.usage.total_tokens or 0

        return {
            "content": content.strip(),
            "model": response.model,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
            "cost": self._calculate_cost(model, response.usage),
            "finish_reason": choice.finish_reason,
        }

    @staticmethod
    def _calculate_cost(model: str, usage) -> float:
        prompt_cost_per_m, completion_cost_per_m = MODEL_PRICING.get(
            model, (0.0, 0.0)
        )
        if usage is None:
            return 0.0

        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0

        return (prompt_tokens / 1_000_000) * prompt_cost_per_m + (
            completion_tokens / 1_000_000
        ) * completion_cost_per_m
