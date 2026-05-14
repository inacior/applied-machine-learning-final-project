#!/usr/bin/env python3
"""Evaluate previously-saved benchmark answers using a verifier LLM."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from bench.openrouter import OpenRouterClient
from bench.evaluator import LLMEvaluator
from bench.config import VERIFIER_MODEL, OPENROUTER_API_KEY, SMALL_MODELS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark answers using a verifier LLM"
    )

    parser.add_argument(
        "--results-file", "-r",
        type=str,
        required=True,
        help="Path to the answers CSV produced by run_benchmark.py",
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="Verifier model to use (default: gpt-4o-mini). Can be a shortcut (gpt-4o-mini, gpt-4o) or full OpenRouter model ID.",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (overrides OPENROUTER_API_KEY env var)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save evaluated CSV (default: overwrites input file)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock evaluation — marks all answers as CORRECT without API calls",
    )

    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else results_path

    # Resolve model
    model_id = _resolve_verifier(args.model)

    df = pd.read_csv(results_path)

    for col in ("evaluation", "evaluation_explanation", "evaluation_model"):
        if col in df.columns:
            df[col] = df[col].astype(object)
        else:
            df[col] = ""

    required_cols = {"question_number", "question", "ground_truth", "model_answer"}
    missing = required_cols - set(df.columns)

    if missing:
        print(f"❌ Results file is missing required columns: {missing}")
        sys.exit(1)

    print(f"Evaluating: {results_path}")
    print(f"Model:      {model_id}")
    print(f"Rows:       {len(df)}")
    if args.dry_run:
        print("Mode:       DRY RUN (mock evaluations)")
    print()

    if args.dry_run:
        for i, _ in df.iterrows():
            df.at[i, "evaluation"] = "CORRECT"
            df.at[i, "evaluation_explanation"] = "[DRY RUN] Mock evaluation"
        df.to_csv(output_path, index=False)
        print(f"📁 Saved: {output_path}")
        print("✅ Done (dry run).")
        return

    api_key = args.api_key or OPENROUTER_API_KEY
    if not api_key:
        print("❌ No API key found.")
        print("   Set it in HW3/.env:  OPENROUTER_API_KEY=sk-or-v1-...")
        print("   Or pass via:          --api-key sk-or-v1-...")
        sys.exit(1)

    client = OpenRouterClient(api_key=api_key)
    evaluator = LLMEvaluator(client, model=model_id)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        if row.get("evaluation") and str(row["evaluation"]) not in ("", "nan", "ERROR"):
            continue

        model_answer = str(row["model_answer"])
        if model_answer.startswith("ERROR:") or model_answer.startswith("[DRY RUN]"):
            df.at[i, "evaluation"] = "SKIPPED"
            df.at[i, "evaluation_explanation"] = "Answer was an error or dry run"
            continue

        observations = row.get("observations", "")
        if pd.isna(observations):
            observations = ""

        try:
            result = evaluator.evaluate(
                question=str(row["question"]),
                ground_truth=str(row["ground_truth"]),
                observations=str(observations),
                model_answer=model_answer,
            )
            df.at[i, "evaluation"] = result["classification"]
            df.at[i, "evaluation_explanation"] = result["explanation"]
            df.at[i, "evaluation_model"] = model_id

        except Exception as exc:
            print(f"\n  ❌ Q{row['question_number']} evaluation failed: {exc}")
            df.at[i, "evaluation"] = "EVAL_ERROR"
            df.at[i, "evaluation_explanation"] = str(exc)

    df.to_csv(output_path, index=False)

    # Summary
    valid = df[~df["evaluation"].isin(["", "ERROR", "SKIPPED", "EVAL_ERROR"])]
    if len(valid) > 0:
        total = len(valid)
        correct = (valid["evaluation"] == "CORRECT").sum()
        partial = (valid["evaluation"] == "PARTIALLY_CORRECT").sum()
        incorrect = (valid["evaluation"] == "INCORRECT").sum()

        print(f"\n{'=' * 50}")
        print(f"  EVALUATION RESULTS")
        print(f"{'=' * 50}")
        print(f"  Evaluated:        {total}")
        print(f"  CORRECT:          {correct} ({correct / total * 100:.1f}%)")
        print(f"  PARTIALLY_CORRECT:{partial} ({partial / total * 100:.1f}%)")
        print(f"  INCORRECT:        {incorrect} ({incorrect / total * 100:.1f}%)")
        print(f"{'=' * 50}")

    print(f"\n📁 Saved: {output_path}")
    print("✅ Done.")


def _resolve_verifier(model_arg: str) -> str:
    shortcuts = {
        "gpt-4o-mini": VERIFIER_MODEL,
        **SMALL_MODELS,
    }
    return shortcuts.get(model_arg, model_arg)


if __name__ == "__main__":
    main()
