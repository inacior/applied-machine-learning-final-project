import sys
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bench.data_loader import (
    load_questions,
    load_dataset,
    format_full_dataset,
    format_spacy_dataset,
)
from bench.ner_pipeline import get_ner_context
from bench.prompts import ANSWER_SYSTEM_PROMPT
from bench.openrouter import OpenRouterClient


_RESULT_COLUMNS = [
    "question_number",
    "classification",
    "model_name",
    "question",
    "ground_truth",
    "observations",
    "csv_row_indices",
    "model_answer",
    "evaluation",
    "evaluation_explanation",
    "tokens_prompt",
    "tokens_completion",
    "tokens_total",
    "cost",
    "latency_ms",
    "dataset_path",
    "questions_path",
]


def run_questions(
    model_name: str,
    model_id: str,
    dataset_path: str | Path,
    questions_path: str | Path,
    output_dir: str | Path,
    api_key: str | None = None,
    max_questions: int | None = None,
    dry_run: bool = False,
    question_filter: set[int] | None = None,
    method: str = "baseline",
) -> pd.DataFrame:
    questions_df = load_questions(questions_path)

    if method == "spacy":
        spacy_path = Path("data/thea_ner_augmented.csv")
        if not spacy_path.exists():
            print(
                "❌ Spacy augmented dataset not found: data/thea_ner_augmented.csv\n"
                "   Run: python ner_spacy.py"
            )
            sys.exit(1)
        dataset_df = load_dataset(spacy_path)
        dataset_text = format_spacy_dataset(dataset_df)
        dataset_intro = (
            "Here is the dataset with spaCy NER annotations:\n\n"
        )
    else:
        dataset_df = load_dataset(dataset_path)
        dataset_text = format_full_dataset(dataset_df)
        dataset_intro = "Here is the complete dataset:\n\n"

    if max_questions is not None:
        questions_df = questions_df.head(max_questions)

    if question_filter is not None:
        questions_df = questions_df[
            questions_df["question_number"].isin(question_filter)
        ]

    results: list[dict] = []
    rows = list(questions_df.iterrows())

    if dry_run:
        for _, row in tqdm(rows, total=len(rows), desc=f"{model_name}"):
            results.append(
                _dry_result(row, model_name, dataset_path, questions_path)
            )
        return _save(df_out := pd.DataFrame(results, columns=_RESULT_COLUMNS), output_dir, model_name)

    client = OpenRouterClient(api_key=api_key)
    benchmark_start = time.time()

    if method == "langextract":
        cache_path = Path(output_dir) / "_ner_cache" / "extraction.json"
        ner_summary = get_ner_context(dataset_df, cache_path=cache_path)
        dataset_text = (
            "# Structured Dataset Overview\n\n"
            "Use this overview as a navigation guide to help you locate relevant "
            "information in the full dataset below.\n\n"
            f"{ner_summary}\n"
            "---\n\n"
            "# Complete Dataset\n\n"
            f"{dataset_text}"
        )
        dataset_intro = (
            "Here is the dataset with a structured navigation guide:\n\n"
        )

    for _, row in tqdm(rows, total=len(rows), desc=f"{model_name}"):
        question_num = row["question_number"]
        question = row["question"]
        ground_truth = row["answer"]
        row_indices = row["csv_row_indices"]

        try:
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"{dataset_intro}{dataset_text}"},
                {"role": "user", "content": (
                    f"## Question\n\n{question}\n\n"
                    f"Answer the question based solely on the dataset above.\n\n"
                    f"Rules reminder:\n"
                    f"- Answer concisely and precisely, drawing ONLY from the provided dataset.\n"
                    f"- Do NOT use any outside knowledge, internet searches, training data, or information beyond what is explicitly provided in the dataset.\n"
                    f"- Do NOT include direct quotes or line references in your answer. Paraphrase and summarize the relevant information in plain prose.\n"
                    f"- Keep your answer concise, but make sure it is complete: do not cut it off mid-thought or mid-sentence. Write as much as needed to fully answer the question.\n"
                    f"- If the dataset does not contain enough information to answer, state that honestly."
                )},
            ]

            question_start = time.time()
            answer = client.chat_with_history(
                model=model_id,
                messages=messages,
            )
            latency_ms = (time.time() - question_start) * 1000

            results.append({
                "question_number": question_num,
                "classification": row["classification"],
                "model_name": model_name,
                "question": question,
                "ground_truth": ground_truth,
                "observations": row.get("observations", ""),
                "csv_row_indices": str(row_indices),
                "model_answer": answer["content"],
                "evaluation": "",
                "evaluation_explanation": "",
                "tokens_prompt": answer["tokens_prompt"],
                "tokens_completion": answer["tokens_completion"],
                "tokens_total": answer["tokens_total"],
                "cost": answer["cost"],
                "latency_ms": latency_ms,
                "dataset_path": str(dataset_path),
                "questions_path": str(questions_path),
            })

        except Exception as exc:
            print(f"\n  ❌ Q{question_num} failed: {exc}")
            results.append(
                _error_result(row, model_name, dataset_path, questions_path, str(exc))
            )

    return _save(pd.DataFrame(results, columns=_RESULT_COLUMNS), output_dir, model_name)


def _dry_result(row, model_name: str, dataset_path, questions_path) -> dict:
    return {
        "question_number": row["question_number"],
        "classification": row["classification"],
        "model_name": model_name,
        "question": row["question"],
        "ground_truth": row["answer"],
        "observations": row.get("observations", ""),
        "csv_row_indices": str(row.get("csv_row_indices", "")),
        "model_answer": f"[DRY RUN] Mock answer for Q{row['question_number']}",
        "evaluation": "",
        "evaluation_explanation": "",
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "cost": 0.0,
        "latency_ms": 0.0,
        "dataset_path": str(dataset_path),
        "questions_path": str(questions_path),
    }


def _error_result(row, model_name: str, dataset_path, questions_path, error_msg: str) -> dict:
    return {
        "question_number": row["question_number"],
        "classification": row["classification"],
        "model_name": model_name,
        "question": row["question"],
        "ground_truth": row["answer"],
        "observations": row.get("observations", ""),
        "csv_row_indices": str(row.get("csv_row_indices", "")),
        "model_answer": f"ERROR: {error_msg}",
        "evaluation": "ERROR",
        "evaluation_explanation": error_msg,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "cost": 0.0,
        "latency_ms": 0.0,
        "dataset_path": str(dataset_path),
        "questions_path": str(questions_path),
    }


def _save(df: pd.DataFrame, output_dir: str | Path, model_name: str) -> pd.DataFrame:
    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers.csv"
    df.to_csv(path, index=False)
    print(f"\n📁 Answers saved: {path}")
    return df
