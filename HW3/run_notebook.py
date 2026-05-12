#!/usr/bin/env python3
"""Extracts and runs the preprocessing cells from the HW2 notebook, saves final df as CSV."""

import os, sys

# --- Cell 4: Setup ---
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("umerhaddii/shakespeare-plays-dialogues")
print("Path to dataset files:", path)

rj_path = os.path.join(path, "romeo_juliet.csv")
df = pd.read_csv(rj_path)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# --- Cell 6: add_participants ---
import re
import html
from typing import Optional

def add_participants(df: pd.DataFrame, direction_col='character', dialogue_col='dialogue',
                     act_col='act', scene_col='scene', stage_label='[stage direction]') -> pd.DataFrame:
    speakers = df.loc[df[direction_col] != stage_label, direction_col].unique()
    char_map = {}
    for c in speakers:
        key = c.strip().upper()
        char_map[key] = c
        if key.startswith('THE '):
            char_map[key[4:]] = c

    def _clean(text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        for pat in [r'^[aA]\s+', r'^[tT][hH][eE]\s+', r'^[aA][nN][dD]\s+',
                    r'^[wW][iI][tT][hH]\s+', r'^[hH][iI][sS]\s+', r'^[hH][eE][rR]\s+']:
            text = re.sub(pat, '', text)
        text = re.sub(r',.*$', '', text)
        text = re.sub(r'\s+(of|with|in|armed|bearing|meeting|booted|disguised|asleep|who)\s+.*$', '',
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\s+(above|below|within|without|booted|disguised|asleep)$', '',
                      text, flags=re.IGNORECASE)
        return text.strip()

    def _extract(text: str) -> list[str]:
        names = []
        for part in re.split(r',\s*|;\s*|\s+and\s+', text):
            cleaned = _clean(part)
            if cleaned and cleaned.upper() in char_map:
                names.append(char_map[cleaned.upper()])
        return names

    def _process_direction(dial: str, present: list, last_speaker: Optional[str]) -> list:
        d = html.unescape(str(dial).strip())
        du = d.upper()

        m = re.search(r'\b(ENTER|RE-ENTER)\b', du)
        if m:
            after = d[m.end():].strip()
            after = re.sub(r'^,\s*', '', after)
            for n in _extract(after):
                if n not in present:
                    present.append(n)
            return present

        if du.startswith('EXEUNT'):
            after = re.sub(r'^EXEUNT\s*', '', d, flags=re.IGNORECASE).strip()
            if not after or after in ('.', ','):
                present = []
            elif re.match(r'^all\s+but\s+', after, flags=re.IGNORECASE):
                after = re.sub(r'^all\s+but\s+', '', after, flags=re.IGNORECASE)
                keep = _extract(after)
                present = [n for n in keep if n in present]
            else:
                for n in _extract(after):
                    if n in present:
                        present.remove(n)
            return present

        if du.startswith('EXIT'):
            after = re.sub(r'^EXIT\s*', '', d, flags=re.IGNORECASE).strip()
            if not after or after in ('.', ','):
                if last_speaker and last_speaker in present:
                    present.remove(last_speaker)
            else:
                for n in _extract(after):
                    if n in present:
                        present.remove(n)
            return present

        return present

    participants = []
    present = []
    last_speaker = None
    cur_act, cur_scene = None, None

    for _, row in df.iterrows():
        act, scene = row[act_col], row[scene_col]
        char = row[direction_col]
        dial = row[dialogue_col]

        if (act, scene) != (cur_act, cur_scene):
            present = []
            last_speaker = None
            cur_act, cur_scene = act, scene

        if char == stage_label:
            present = _process_direction(dial, present, last_speaker)
        else:
            if char not in present:
                present.append(char)
            last_speaker = char

        participants.append(list(present))

    return df.assign(participants=participants)

df = add_participants(df)
print(f"After add_participants: {df.shape}, columns: {list(df.columns)}")

# --- Cell 8: normalize_character ---
TITLES = {
    "MR", "MRS", "MS", "DR", "SIR", "LORD", "LADY",
    "PRINCE", "KING", "QUEEN", "DUKE", "DUCHESS",
    "FRIAR", "CAPTAIN", "SERVANT", "NURSE"
}

def normalize_character(name: str):
    if not isinstance(name, str):
        return {
            "title": None,
            "first_name": None,
            "last_name": None,
            "normalized_name": None
        }

    name = name.strip().upper()
    name = re.sub(r'\s+', ' ', name)

    parts = name.split()

    title = None
    first_name = None
    last_name = None

    if parts[0] in TITLES:
        title = parts[0]
        remaining = parts[1:]

        if len(remaining) == 0:
            first_name = title
        elif len(remaining) == 1:
            last_name = remaining[0]
        else:
            first_name = remaining[0]
            last_name = " ".join(remaining[1:])
    else:
        if len(parts) == 1:
            first_name = parts[0]
        elif len(parts) == 2:
            first_name, last_name = parts
        else:
            first_name = parts[0]
            last_name = " ".join(parts[1:])

    normalized = " ".join([p for p in [first_name, last_name] if p])

    return {
        "title": title,
        "first_name": first_name,
        "last_name": last_name,
        "normalized_name": normalized
    }

parsed = df['character'].apply(normalize_character)

df['title'] = parsed.apply(lambda x: x['title'])
df['first_name'] = parsed.apply(lambda x: x['first_name'])
df['last_name'] = parsed.apply(lambda x: x['last_name'])
df['normalized_name'] = parsed.apply(lambda x: x['normalized_name'])

print(f"After normalize_character: {df.shape}, columns: {list(df.columns)}")

# --- Cell 10: merge_split_dialogues ---
def merge_split_dialogues(df):
    merged_rows = []

    current_row = None

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        if current_row is None:
            current_row = row_dict
            continue

        same_character = row_dict['character'] == current_row['character']
        same_scene = (
            row_dict['act'] == current_row['act'] and
            row_dict['scene'] == current_row['scene']
        )

        try:
            prev_line = int(current_row['line_number'])
            curr_line = int(row_dict['line_number'])
            consecutive = curr_line == prev_line + 1
        except:
            consecutive = False

        is_stage = row_dict['character'] == '[stage direction]'
        prev_is_stage = current_row['character'] == '[stage direction]'

        if same_character and same_scene and consecutive and not is_stage and not prev_is_stage:
            current_row['dialogue'] += ' ' + str(row_dict['dialogue'])
            current_row['line_number'] = row_dict['line_number']
        else:
            merged_rows.append(current_row)
            current_row = row_dict

    if current_row is not None:
        merged_rows.append(current_row)

    return pd.DataFrame(merged_rows)


df = merge_split_dialogues(df)
df = df.reset_index(drop=True)

print(f"After merge_split_dialogues: {df.shape}, columns: {list(df.columns)}")

# --- Save final CSV ---
os.makedirs("data", exist_ok=True)
df.to_csv("data/romeo_juliet_preprocessed.csv", index=False)
print(f"Saved {df.shape[0]} rows, {df.shape[1]} columns to data/romeo_juliet_preprocessed.csv")
print(df.head(3))
