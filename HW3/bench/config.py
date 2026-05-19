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

VERIFIER_MODEL = "openai/gpt-5.4-mini"

SMALL_MODELS: dict[str, str] = {
    "gemma-3-4b":     "google/gemma-3-4b-it",
    "llama-1b":       "meta-llama/llama-3.2-1b-instruct",
    "llama-3b":       "meta-llama/llama-3.2-3b-instruct",
    "ministral-3b":   "mistralai/ministral-3b-2512",
    "phi-4-mini":     "microsoft/phi-4-mini-instruct",
    # "qwen-3.5-9b":    "qwen/qwen3.5-9b",
    # "qwen-3.5-27b":   "qwen/qwen3.5-27b",
    "mistral-small-3.2": "mistralai/mistral-small-3.2-24b-instruct-2506",
    "qwen3-30b-a3b":  "qwen/qwen3-30b-a3b-instruct-2507",
    # "mistral-7b":     "mistralai/mistral-7b-instruct-v0.1",  # ~3K context — too small for full dataset
    # "mistral-saba":   "mistralai/mistral-saba",              # 32K context — too small for full dataset
    # "qwen-2.5-7b":    "qwen/qwen-2.5-7b-instruct",          # 32K context — too small for full dataset
}

# ---------------------------------------------------------------------------
# Rate Limiting & Retry
# ---------------------------------------------------------------------------
RATE_LIMIT_DELAY = 1.0       # seconds between consecutive API calls
MAX_RETRIES = 3              # total attempts (1 initial + 2 retries)
RETRY_BACKOFF = 1.0          # fixed wait between retries (seconds)
REQUEST_TIMEOUT = 60        # seconds before HTTP timeout

# ---------------------------------------------------------------------------
# NER Pipeline Configuration
# ---------------------------------------------------------------------------
NER_MODEL_ID = "openrouter/deepseek/deepseek-v4-flash"
NER_EXTRACTION_PASSES = 2
NER_MAX_WORKERS = 10
NER_CACHE_DIR = HW3_DIR / ".ner_cache"

# Enable NER for small models by default (can override via CLI)
USE_NER_FOR_SMALL_MODELS = True

# ---------------------------------------------------------------------------
# Generation Parameters
# ---------------------------------------------------------------------------
MAX_TOKENS_ANSWER = 2048     # max tokens for answering LLM
MAX_TOKENS_VERIFY = 2048     # max tokens for verification LLM
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
    "qwen/qwen-2.5-7b-instruct":              (0.07,  0.07),
    "qwen/qwen3.5-9b":                        (0.04,  0.15),
    "openai/gpt-oss-120b":                    (0.039,  0.18),
    "openai/gpt-5.4":                         (2.50,  15.00),
}
