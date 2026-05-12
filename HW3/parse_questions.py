#!/usr/bin/env python3
"""Parse questions_forensics.md and save as CSV."""

import re
import csv
import os

with open("../romeo-juliet-dump/questions_forensics.md") as f:
    content = f.read()

# Split by question headers
entries = re.split(r'\n## Question #(\d+)\n', content)
# first element is the preamble (title + intro text)
# entries[0] = preamble, entries[1] = "1", entries[2] = q1 body, entries[3] = "2", etc.

rows = []
i = 1
while i < len(entries):
    q_num = int(entries[i])
    body = entries[i + 1]
    i += 2

    # Extract classification
    classification_match = re.search(r'\*\*Classification:\*\*\s*(\w+)', body)
    classification = classification_match.group(1) if classification_match else ""

    # Extract question (text after "**Question:**" until next bold field or Observation)
    question_match = re.search(r'\*\*Question:\*\*\s*(.+?)(?=\n\n\*\*|\n> \*\*)', body, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""

    # Extract answer
    answer_match = re.search(r'\*\*Answer:\*\*\s*(.+?)(?=\n\n\*\*|\n> \*\*)', body, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    # Extract observations
    obs_match = re.search(r'\*\*Observations:\*\*\s*(.+?)$', body, re.DOTALL)
    observations = obs_match.group(1).strip() if obs_match else ""

    # Extract line references from observations (e.g., L127-L165, L196-L202)
    line_ranges = re.findall(r'L(\d+)-L(\d+)', observations)
    line_refs_str = "; ".join(f"{s}-{e}" for s, e in line_ranges) if line_ranges else ""

    rows.append({
        "question_number": q_num,
        "classification": classification,
        "question": question,
        "answer": answer,
        "observations": observations,
        "line_start": int(line_ranges[0][0]) if line_ranges else None,
        "line_end": int(line_ranges[-1][1]) if line_ranges else None,
        "line_references_raw": line_refs_str,
    })

os.makedirs("data", exist_ok=True)
csv_path = "data/benchmark_questions.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} questions to {csv_path}")
print(f"Classification distribution: {dict((c, sum(1 for r in rows if r['classification']==c)) for c in ['easy','medium','hard'])}")
