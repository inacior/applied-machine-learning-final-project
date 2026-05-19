#!/usr/bin/env python3
"""Standalone test script for the NER pipeline.

Runs only the LangExtract extraction on the Romeo & Juliet dataset.
Outputs the structured NER text and saves it to a file for inspection.
"""

import time
from pathlib import Path

import sys

# Ensure bench package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.data_loader import load_dataset, format_full_dataset
from bench.ner_pipeline import extract_entities, format_entities_to_context


def main():
    dataset_path = Path("data/romeo_juliet_preprocessed.csv")
    cache_path = Path(".ner_test_cache/extraction.json")

    print(f"Dataset: {dataset_path}")
    print("Loading dataset...")
    df = load_dataset(dataset_path)
    print(f"  Rows: {len(df)}")

    print("Formatting dataset text...")
    dataset_text = format_full_dataset(df)
    print(f"  Text length: {len(dataset_text)} chars")

    print("\n--- Running NER extraction ---")
    start = time.time()
    result = extract_entities(dataset_text, cache_path=cache_path)
    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Extractions: {len(result.extractions)}")

    print("\n--- Formatting to structured context ---")
    context = format_entities_to_context(result)
    print(f"  Context length: {len(context)} chars")

    output_path = Path("ner_output.txt")
    with open(output_path, "w") as f:
        f.write(context)
    print(f"\n  Saved structured output to: {output_path}")

    print("\n--- Preview (first 1500 chars) ---")
    print(context[:1500])


if __name__ == "__main__":
    main()
