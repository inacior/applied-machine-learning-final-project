# HW3 Handoff

## Project Overview

This project investigates whether lightweight, deterministic entity extraction preprocessing can improve the question-answering accuracy of small open-weight language models (2B–7B parameters) on a forensics-style narrative comprehension benchmark using Romeo & Juliet.

### Five-Stage Pipeline

1. **NER Pipeline** — Parse the text to extract characters, relationships, family affiliations, locations, and events. Build a structured entity graph + per-entity profiles.
2. **Benchmark Construction** — 60 QA pairs with difficulty labels (easy, medium, hard), structured as CSV with line-range validation references.
3. **SLM Evaluation via OpenRouter** — Query 2B and 7B models in two configurations: (a) raw text only, (b) entity-augmented context.
4. **Analysis** — Exact/semantic match accuracy, stratified by difficulty and question type. Token-cost comparison. Ablation on entity info types.
5. **Proprietary baseline** — Run the same benchmark on a frontier LLM for the performance ceiling.

**Hypothesis:** Structured entity context will yield the largest gains on identity and relationship questions while offering marginal benefit on purely thematic/interpretive items.

---

## Data Assets in `HW3/data/`

### 1. `romeo_juliet_preprocessed.csv`
- **Source:** Kaggle dataset `umerhaddii/shakespeare-plays-dialogues` → `romeo_juliet.csv`
- **Rows:** 1068 (after merging consecutive same-character lines within a scene)
- **Columns:** 10
  - `act` — Act I through Act V
  - `scene` — Prologue, Scene I–V
  - `character` — Speaker name (e.g., Romeo, Juliet, Lady Capulet) or `[stage direction]`
  - `dialogue` — Merged dialogue text for consecutive lines by the same character in the same scene
  - `line_number` — Last sequential line number in the merged block (from original dataset)
  - `participants` — Python-list of characters physically on stage at that moment
  - `title` — Extracted honorific (e.g., LADY, PRINCE, NURSE)
  - `first_name` — First name component
  - `last_name` — Last name component (e.g., CAPULET, MONTAGUE)
  - `normalized_name` — Canonical name string
- **Key feature:** The `participants` column tracks real-time on-stage presence by parsing Enter/Exit/Exeunt stage directions, enabling interaction analysis.

### 2. `benchmark_questions.csv`
- **Source:** Parsed from `romeo-juliet-dump/questions_forensics.md`
- **Rows:** 60 questions
- **Classification distribution:** easy (11), medium (31), hard (18)
- **Columns:** 9
  - `question_number` — 1 through 60
  - `classification` — easy, medium, or hard
  - `question` — Full question text
  - `answer` — Ground-truth answer
  - `observations` — Full observations text from the source markdown
  - `line_start` — First referenced line number in `raw_dump.md`
  - `line_end` — Last referenced line number in `raw_dump.md`
  - `line_references_raw` — All reference ranges as strings (e.g., `127-165; 196-202`)
  - `csv_row_indices` — Python-list of 0-indexed row numbers in `romeo_juliet_preprocessed.csv` where the answer's source evidence can be verified
- **Verification:** All 60 questions were verified line-by-line. Zero mismatches. Every `csv_row_indices` correctly points to the relevant source text.

---

## Source Files

Located at `../romeo-juliet-dump/`:
- `raw_dump.md` — Complete text of Romeo & Juliet (6106 lines) with speaker-attributed dialogue, stage directions, and scene metadata. Line numbers referenced in `benchmark_questions.csv` refer to lines in this file (1-indexed).
- `questions_forensics.md` — 60 forensics-style QA pairs with difficulty labels and answer validation references
- `questions_literature.md` — 60 literature-style QA pairs (same questions, different phrasing)
- `raw_dump.html` — HTML version of the play text

---

## Project Structure

```
HW1/
  extended_abstract.md          # Project proposal
HW2/
  [Group_5]_HW2_EDA_and_Preprocessing.ipynb  # EDA notebook (produces the preprocessed CSV)
romeo-juliet-dump/
  raw_dump.md                   # Source text (6106 lines)
  questions_forensics.md         # 60 forensics QA pairs
  questions_literature.md        # 60 literature QA pairs
HW3/
  data/
    romeo_juliet_preprocessed.csv   # Preprocessed play text (1068 rows)
    benchmark_questions.csv          # Structured benchmark (60 rows)
```

---

## Key Mappings

- The `line_start`/`line_end` columns in `benchmark_questions.csv` reference 1-indexed line numbers in `raw_dump.md` (the file as viewed in a markdown reader).
- The `csv_row_indices` column maps those raw dump line ranges to 0-indexed rows in `romeo_juliet_preprocessed.csv`.
- Example: Q1 (thumb-biting incident) references dump lines 127-202 → maps to CSV rows [24-42], which contain Sampson's "I will bite my thumb at them" and his denial.

---

## Step Implemented
1. **NER Pipeline** — Extract named entities (characters, families, locations) from the preprocessed CSV to build entity profiles and a relationship graph.

# pip install -r requirements.txt
# python -m spacy download en_core_web_sm
# pip install pyvis==0.1.9`
# python ner_graph_extract.py

## Next Steps (Not Yet Implemented)


2. **Entity-Augmented Context** — For each question, prepend structured entity annotations to the source text before sending to the LLM.
3. **OpenRouter Integration** — Query 2B and 7B models in raw and entity-augmented modes.
4. **Evaluation** — Compute exact-match and semantic-match accuracy, stratified by difficulty and question type.
5. **Baseline** — Run the same benchmark against a proprietary frontier LLM.
