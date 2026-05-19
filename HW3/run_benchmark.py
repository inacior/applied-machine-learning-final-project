#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from bench.config import BASELINE_MODEL, SMALL_MODELS, OPENROUTER_API_KEY, USE_NER_FOR_SMALL_MODELS
from bench.runner import run_questions


def _resolve_model(model_arg: str) -> tuple[str, str]:
    if model_arg == "baseline":
        return "baseline-gpt4o", BASELINE_MODEL
    if model_arg in SMALL_MODELS:
        return model_arg, SMALL_MODELS[model_arg]
    return model_arg, model_arg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QA benchmark — asks an LLM questions based on a dataset CSV"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="baseline",
        choices=["baseline", "gemma-3-4b", "llama-1b", "llama-3b", "ministral-3b", "phi-4-mini", "qwen-3.5-9b", "qwen-3.5-27b", "mistral-small-3.2", "qwen3-30b-a3b"],
        help="Model to query (default: baseline = GPT-4o)",
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="data/romeo_juliet_preprocessed.csv",
        help="Path to the dataset CSV (default: data/romeo_juliet_preprocessed.csv)",
    )

    parser.add_argument(
        "--questions", "-q",
        type=str,
        default="data/benchmark_questions.csv",
        help="Path to the questions CSV (default: data/benchmark_questions.csv)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Directory to save answers CSV (default: results)",
    )

    parser.add_argument(
        "--max-questions", "-n",
        type=int,
        default=None,
        help="Limit to first N questions (default: all)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test the pipeline without API calls — generates mock answers",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (overrides OPENROUTER_API_KEY env var)",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Comma-separated question numbers to run (e.g. '1,5,10-15')",
    )

    parser.add_argument(
        "--use-ner",
        action="store_true",
        help="Use NER-extracted structured context instead of raw CSV for small models",
    )

    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Force raw CSV context even for small models (override default)",
    )

    args = parser.parse_args()

    model_name, model_id = _resolve_model(args.model)

    dataset_path = Path(args.dataset)
    questions_path = Path(args.questions)
    output_dir = Path(args.output)

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    if not questions_path.exists():
        print(f"❌ Questions file not found: {questions_path}")
        sys.exit(1)

    print(f"Model:     {model_name} ({model_id})")
    print(f"Dataset:   {dataset_path}")
    print(f"Questions: {questions_path}")
    print(f"Output:    {output_dir}")
    if args.dry_run:
        print("Mode:      DRY RUN (no API calls)")
    print()

    if not args.dry_run:
        api_key = args.api_key or OPENROUTER_API_KEY
        if not api_key:
            print("❌ No API key found.")
            print("   Set it in HW3/.env:  OPENROUTER_API_KEY=sk-or-v1-...")
            print("   Or pass via:          --api-key sk-or-v1-...")
            print("   Or test pipeline via: --dry-run")
            sys.exit(1)
    else:
        api_key = None

    question_filter = None
    if args.filter:
        question_filter = set()
        for part in args.filter.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                question_filter.update(range(int(start), int(end) + 1))
            else:
                question_filter.add(int(part))

    use_ner = False
    if args.use_ner:
        use_ner = True
    elif args.no_ner:
        use_ner = False
    elif model_name in SMALL_MODELS and USE_NER_FOR_SMALL_MODELS:
        use_ner = True

    if use_ner:
        print("Mode:      NER-enhanced context for small models")

    run_questions(
        model_name=model_name,
        model_id=model_id,
        dataset_path=dataset_path,
        questions_path=questions_path,
        output_dir=output_dir,
        api_key=api_key,
        max_questions=args.max_questions,
        dry_run=args.dry_run,
        question_filter=question_filter,
        use_ner=use_ner,
    )

    print("✅ Done. Run evaluate_results.py to score the answers.")


if __name__ == "__main__":
    main()
