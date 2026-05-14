import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bench.data_loader import load_questions, load_dataset, get_context_rows, format_rows
from bench.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
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
) -> pd.DataFrame:
    questions_df = load_questions(questions_path)
    dataset_df = load_dataset(dataset_path)

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
        return _save(df_out := pd.DataFrame(results, columns=_RESULT_COLUMNS), output_dir)

    client = OpenRouterClient(api_key=api_key)
    benchmark_start = time.time()

    for _, row in tqdm(rows, total=len(rows), desc=f"{model_name}"):
        question_num = row["question_number"]
        question = row["question"]
        ground_truth = row["answer"]
        row_indices = row["csv_row_indices"]

        try:
            context_rows = get_context_rows(dataset_df, row_indices)
            context_text = format_rows(context_rows)
            user_message = ANSWER_USER_TEMPLATE.format(
                context=context_text, question=question
            )

            answer = client.chat(
                model=model_id,
                system_prompt=ANSWER_SYSTEM_PROMPT,
                user_message=user_message,
            )

            latency_ms = (time.time() - benchmark_start) * 1000

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

    return _save(pd.DataFrame(results, columns=_RESULT_COLUMNS), output_dir)


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


def _save(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers.csv"
    df.to_csv(path, index=False)
    print(f"\n📁 Answers saved: {path}")
    return df
