import ast
import pandas as pd
from pathlib import Path


def load_questions(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["csv_row_indices"] = df["csv_row_indices"].apply(_safe_parse_list)
    return df


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_context_rows(dataset_df: pd.DataFrame, row_indices: list[int]) -> pd.DataFrame:
    return dataset_df.iloc[row_indices]


def format_rows(rows_df: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in rows_df.iterrows():
        character = str(row.get("character", ""))
        dialogue = str(row.get("dialogue", ""))
        act = str(row.get("act", ""))
        scene = str(row.get("scene", ""))
        location = f"{act}, {scene}"

        if character == "[stage direction]":
            lines.append(f"[{location}] *{dialogue}*")
        else:
            lines.append(f"[{location}] {character}: \"{dialogue}\"")
    return "\n".join(lines)


def format_full_dataset(dataset_df: pd.DataFrame) -> str:
    lines = ["line_number | act | scene | character | dialogue"]
    for _, row in dataset_df.iterrows():
        line_num_val = row.get("line_number")
        if pd.notna(line_num_val):
            try:
                line_num = str(int(float(line_num_val)))
            except (ValueError, TypeError):
                line_num = str(line_num_val)
        else:
            line_num = ""
        act = str(row.get("act", ""))
        scene = str(row.get("scene", ""))
        character = str(row.get("character", ""))
        dialogue = str(row.get("dialogue", ""))
        lines.append(f"{line_num} | {act} | {scene} | {character} | {dialogue}")
    return "\n".join(lines)


def _safe_parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or not value:
        return []
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
