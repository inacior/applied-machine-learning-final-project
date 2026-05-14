"""Configuration: environment variables, paths, model registry, and API settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from HW3/ directory (parent of bench/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HW3_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = HW3_DIR / "data"
RESULTS_DIR = HW3_DIR / "results"

# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

# Frontier baseline — highest-capability model for the performance ceiling
BASELINE_MODEL = "openai/gpt-5.4"

VERIFIER_MODEL = "openai/gpt-5.4"

SMALL_MODELS: dict[str, str] = {
    "llama-3b":        "meta-llama/llama-3.2-3b-instruct",
    "mistral-7b":      "mistralai/mistral-7b-instruct-v0.1",
    "ministral-3b":    "mistralai/ministral-3b-2512",
    "mistral-saba":    "mistralai/mistral-saba",
    "gemma-3-4b":      "google/gemma-3-4b-it",
    "qwen-2.5-7b":     "qwen/qwen-2.5-7b-instruct",
}

# ---------------------------------------------------------------------------
# Rate Limiting & Retry
# ---------------------------------------------------------------------------
RATE_LIMIT_DELAY = 1.0       # seconds between consecutive API calls
MAX_RETRIES = 3              # total attempts (1 initial + 2 retries)
RETRY_BACKOFF = 1.0          # fixed wait between retries (seconds)
REQUEST_TIMEOUT = 20        # seconds before HTTP timeout

# ---------------------------------------------------------------------------
# Generation Parameters
# ---------------------------------------------------------------------------
MAX_TOKENS_ANSWER = 512      # max tokens for answering LLM
MAX_TOKENS_VERIFY = 256      # max tokens for verification LLM
TEMPERATURE = 0.0            # deterministic sampling for reproducibility

# ---------------------------------------------------------------------------
# Expected cost per million tokens (USD): (prompt, completion)
# Used for cost estimation from usage data.
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "meta-llama/llama-3.2-3b-instruct":        (0.015,  0.02),
    "mistralai/mistral-7b-instruct-v0.1":     (0.11,   0.19),
    "mistralai/ministral-3b-2512":            (0.10,   0.10),
    "mistralai/mistral-saba":                 (0.20,   0.60),
    "google/gemma-3-4b-it":                   (0.04,   0.04),
    "google/gemma-4-26b-a4b-it:free":         (0.00,   0.00),
    "google/gemma-4-31b-it:free":             (0.00,   0.00),
    "qwen/qwen-2.5-7b-instruct":              (0.07,   0.07),
    "openai/gpt-oss-120b":                    (0.039,  0.18),
    "openai/gpt-5.4":                         (2.50,  15.00),
}
