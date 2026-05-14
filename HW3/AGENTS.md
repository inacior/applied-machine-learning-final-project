# HW3 — Benchmark Evaluation System

## Setup

```bash
cd HW3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Paste your OpenRouter API key in `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

Get a key at: https://openrouter.ai/keys

---

## Architecture

Two-step pipeline: **answer** → **evaluate**

### Step 1: `run_benchmark.py` — Ask an LLM questions

Takes a dataset CSV + questions CSV, extracts relevant rows per question, sends them to an LLM, saves answers.

```
Dataset CSV  ──┐
               ├──> format rows as text ──> LLM ──> answers CSV
Questions CSV ─┘
```

### Step 2: `evaluate_results.py` — Score the answers

Takes the answers CSV from step 1, sends each answer + ground truth to a verifier LLM, classifies as CORRECT / PARTIALLY_CORRECT / INCORRECT.

```
Answers CSV ──> verifier LLM ──> evaluated CSV (same file, evaluation columns filled)
```

---

## Usage

### Step 1 — Run questions

```bash
source .venv/bin/activate

# Default: baseline (GPT-5.4), existing dataset and questions
python run_benchmark.py --model baseline

# Small model with custom dataset and questions
python run_benchmark.py --model llama-3b \
    --dataset data/romeo_juliet_preprocessed.csv \
    --questions data/benchmark_questions.csv

# Limit to first 10 questions
python run_benchmark.py --model gemma-31b --max-questions 10

# Specific questions only
python run_benchmark.py --model mistral-7b --filter 1,5,10-15

# Dry run (no API calls)
python run_benchmark.py --dry-run --max-questions 5

# Custom output directory
python run_benchmark.py --model llama-3b --output results/my_experiment
```

### Step 2 — Evaluate answers

```bash
# Evaluate with default verifier (GPT-5.4)
python evaluate_results.py --results-file results/answers.csv

# Evaluate with a specific verifier model (shortcut or full ID)
python evaluate_results.py --results-file results/answers.csv --model gpt-5.4
python evaluate_results.py --results-file results/answers.csv --model openai/gpt-5.4

# Dry run (mock evaluations)
python evaluate_results.py --dry-run --results-file results/answers.csv
```

**Safe to re-run**: `evaluate_results.py` skips rows that already have an evaluation. You can run it multiple times on the same file — only unanswered/empty rows are processed.

### Using custom datasets

Later, team members will provide their own pre-processed CSVs. Point `--dataset` at the new file:

```bash
python run_benchmark.py --model llama-3b --dataset data/pedro_ner_augmented.csv
python run_benchmark.py --model mistral-7b --dataset data/thea_ner_augmented.csv
```

Questions CSV stays the same (`data/benchmark_questions.csv`).

---

## Models

| `--model` flag | Model ID | Notes |
|---|---|---|
| `baseline` | `openai/gpt-5.4` | Frontier performance ceiling |
| `llama-3b` | `meta-llama/llama-3.2-3b-instruct` | 3B — cheapest, fast |
| `mistral-7b` | `mistralai/mistral-7b-instruct-v0.1` | 7B — Mistral classic |
| `ministral-3b` | `mistralai/ministral-3b-2512` | 3B — newest Mistral tiny |
| `mistral-saba` | `mistralai/mistral-saba` | 24B — regional model |
| `gemma-3-4b` | `google/gemma-3-4b-it` | 4B — Google edge model |
| `qwen-2.5-7b` | `qwen/qwen-2.5-7b-instruct` | 7B — best all-rounder |

Default verifier: `openai/gpt-5.4` (override with `--model` on evaluate command).

### Free tier caveats

Free-tier models (`:free` suffix) are **slow** and heavily rate-limited:
- OpenRouter limits free models to ~16 requests/minute
- Upstream providers (Google AI Studio) may throttle independently
- Expect **30–120 seconds per question** on Gemma free models
- The system auto-retries with exponential backoff (up to 5 attempts), but free models may still fail under heavy throttling
- For full 60-question runs, use paid models or run in batches with pauses between them

---

## Output format

`results/answers.csv` (after step 1):

| Column | Description |
|---|---|
| `question_number` | 1–60 |
| `classification` | easy / medium / hard |
| `model_name` | Model identifier |
| `question` | Full question text |
| `ground_truth` | Expected answer |
| `observations` | Supporting evidence |
| `model_answer` | LLM's response |
| `evaluation` | *(empty — filled by step 2)* |
| `evaluation_explanation` | *(empty — filled by step 2)* |
| `tokens_prompt` | Prompt tokens |
| `tokens_completion` | Completion tokens |
| `tokens_total` | Total tokens |
| `cost` | Cost in USD |
| `latency_ms` | Cumulative latency |
| `dataset_path` | Source dataset CSV |
| `questions_path` | Source questions CSV |

After step 2, `evaluation` and `evaluation_explanation` are filled. Possible evaluation values: `CORRECT`, `PARTIALLY_CORRECT`, `INCORRECT`, `ERROR` (API failure), `EVAL_ERROR` (verifier failure), `SKIPPED` (already evaluated or error answer).

---

## Rate Limiting & Retries

Configured in `bench/config.py`:

| Setting | Default | Description |
|---|---|---|
| `RATE_LIMIT_DELAY` | 6.0s | Minimum gap between API calls (10/min — safe under 16/min free limit) |
| `MAX_RETRIES` | 5 | Total attempts (1 initial + 4 retries) |
| `RETRY_BACKOFF` | 2.0 | Exponential backoff base (2^n seconds) |
| `REQUEST_TIMEOUT` | 120s | HTTP timeout per request |

On 429 (rate limit) errors, the system waits 15s + 10s per retry attempt instead of the standard backoff.

---

## Prompts

The prompts are generic — no mention of Romeo & Juliet or Shakespeare. The LLM is framed as an **investigator** who must answer solely from the provided data, with an explicit rule against using outside knowledge or internet searches.

---

## Project structure

```
HW3/
├── .env                          # API key (gitignored)
├── requirements.txt
├── run_benchmark.py              # Step 1: ask questions, save answers
├── evaluate_results.py           # Step 2: evaluate answers with verifier LLM
├── bench/
│   ├── __init__.py
│   ├── config.py                 # Model registry, API settings, rate limits, pricing
│   ├── data_loader.py            # CSV loading and row formatting
│   ├── prompts.py                # Investigator + evaluator prompt templates
│   ├── openrouter.py             # API client: retries, rate limiting, cost tracking
│   ├── evaluator.py              # LLM-as-judge classification
│   └── runner.py                 # Question-answering orchestration loop
├── data/
│   ├── romeo_juliet_preprocessed.csv
│   └── benchmark_questions.csv
└── results/                      # Output CSVs (answers + evaluations)
```
