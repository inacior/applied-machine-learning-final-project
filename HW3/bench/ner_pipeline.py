"""NER Pipeline for Romeo & Juliet benchmark.

Uses langextract + OpenRouter (gemini-2.5-flash) to extract structured
entities from the dataset, then formats them for SMALL_MODELS.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import langextract as lx
from bench.config import OPENROUTER_API_KEY

NER_MODEL_ID = "openrouter/google/gemini-2.5-flash"

EXTRACTION_PROMPT = """\
Extract structured information from this Romeo & Juliet play dataset.

Extraction classes:
- character: Named characters.
- relationship: Interactions between characters.
- emotion: Explicit emotional expressions or moods.
- event: Major plot events with act/scene context.
- location: Settings or places.
- object: Significant objects.
- theme: Underlying themes.

Rules:
- Use EXACT text from the source for extraction_text. Do not paraphrase.
- Extract in order of appearance.
- Provide meaningful attributes to add context.
"""

EXAMPLES = [
    lx.data.ExampleData(
        text="14 | Act I | Prologue | Chorus | Two households, both alike in dignity",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Chorus",
            ),
            lx.data.Extraction(
                extraction_class="theme",
                extraction_text="two households, both alike in dignity",
                attributes={"type": "family feud"},
            ),
        ],
    ),
]


@dataclass
class CachedExtraction:
    extraction_class: str
    extraction_text: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedResult:
    extractions: list[CachedExtraction]
    text_preview: str = ""


def extract_entities(
    dataset_text: str,
    cache_path: Path | None = None,
) -> CachedResult:
    """Run langextract on the dataset text. Reads from cache if available."""
    if cache_path and cache_path.exists():
        return _load_from_cache(cache_path)

    print(f"[NER] Extracting entities with {NER_MODEL_ID}...")
    start = time.time()

    result = lx.extract(
        text_or_documents=dataset_text,
        prompt_description=EXTRACTION_PROMPT,
        examples=EXAMPLES,
        model_id=NER_MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        extraction_passes=1,
        max_workers=5,
        max_char_buffer=50000,
    )

    elapsed = time.time() - start
    print(f"[NER] Extracted {len(result.extractions)} entities in {elapsed:.1f}s")

    filtered = [
        e
        for e in result.extractions
        if e.extraction_text in dataset_text
    ]
    if len(filtered) < len(result.extractions):
        print(f"[NER] Filtered {len(result.extractions) - len(filtered)} non-grounded extractions")

    cached = CachedResult(
        extractions=[
            CachedExtraction(
                extraction_class=e.extraction_class,
                extraction_text=e.extraction_text,
                attributes=e.attributes or {},
            )
            for e in filtered
        ],
        text_preview=getattr(result, "text", "")[:500],
    )

    if cache_path:
        _save_to_cache(cache_path, cached)

    return cached


def format_entities_to_context(result: CachedResult) -> str:
    """Convert extracted entities into structured markdown for LLM prompts."""
    groups: dict[str, list[CachedExtraction]] = defaultdict(list)
    for e in result.extractions:
        groups[e.extraction_class].append(e)

    lines = ["# Structured Dataset Overview\n"]

    if "character" in groups:
        lines.append("## Characters")
        chars: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "attrs": defaultdict(set)})
        for e in groups["character"]:
            name = e.extraction_text
            chars[name]["count"] += 1
            for k, v in e.attributes.items():
                chars[name]["attrs"][k].add(str(v))
        for name, data in sorted(chars.items(), key=lambda x: x[1]["count"], reverse=True):
            attr_parts = []
            for k, vals in list(data["attrs"].items())[:3]:
                attr_parts.append(f"{k}: {', '.join(sorted(vals)[:2])}")
            attr_str = f" ({'; '.join(attr_parts)})" if attr_parts else ""
            lines.append(f"- {name}: {data['count']} mentions{attr_str}")
        lines.append("")

    if "relationship" in groups:
        lines.append("## Relationships")
        seen: set[str] = set()
        for e in groups["relationship"]:
            if e.extraction_text not in seen:
                seen.add(e.extraction_text)
                attrs = "; ".join(f"{k}: {v}" for k, v in e.attributes.items())
                lines.append(f"- {e.extraction_text}" + (f" ({attrs})" if attrs else ""))
        lines.append("")

    if "event" in groups:
        lines.append("## Key Events")
        seen = set()
        for e in groups["event"]:
            if e.extraction_text not in seen:
                seen.add(e.extraction_text)
                attrs = "; ".join(f"{k}: {v}" for k, v in e.attributes.items())
                lines.append(f"- {e.extraction_text}" + (f" ({attrs})" if attrs else ""))
        lines.append("")

    if "emotion" in groups:
        lines.append("## Emotional Themes")
        emotions: dict[str, int] = defaultdict(int)
        for e in groups["emotion"]:
            emotions[e.extraction_text] += 1
        for text, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:25]:
            lines.append(f"- {text}: {count} mentions")
        lines.append("")

    if "location" in groups:
        lines.append("## Locations")
        seen = set()
        for e in groups["location"]:
            if e.extraction_text not in seen:
                seen.add(e.extraction_text)
                lines.append(f"- {e.extraction_text}")
        lines.append("")

    if "object" in groups:
        lines.append("## Significant Objects")
        seen = set()
        for e in groups["object"]:
            if e.extraction_text not in seen:
                seen.add(e.extraction_text)
                attrs = "; ".join(f"{k}: {v}" for k, v in e.attributes.items())
                lines.append(f"- {e.extraction_text}" + (f" ({attrs})" if attrs else ""))
        lines.append("")

    if "theme" in groups:
        lines.append("## Themes")
        seen = set()
        for e in groups["theme"]:
            if e.extraction_text not in seen:
                seen.add(e.extraction_text)
                attrs = "; ".join(f"{k}: {v}" for k, v in e.attributes.items())
                lines.append(f"- {e.extraction_text}" + (f" ({attrs})" if attrs else ""))
        lines.append("")

    return "\n".join(lines)


def get_ner_context(
    dataset_df,
    cache_path: Path | None = None,
) -> str:
    """High-level entry point: dataset DataFrame -> structured NER text."""
    from bench.data_loader import format_full_dataset

    dataset_text = format_full_dataset(dataset_df)
    result = extract_entities(dataset_text, cache_path=cache_path)
    return format_entities_to_context(result)


def _save_to_cache(path: Path, result: CachedResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "extractions": [
            {
                "extraction_class": e.extraction_class,
                "extraction_text": e.extraction_text,
                "attributes": e.attributes,

            }
            for e in result.extractions
        ],
        "text_preview": result.text_preview,
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)
    print(f"[NER] Cached to {path}")


def _load_from_cache(path: Path) -> CachedResult:
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[NER] Corrupted cache file, removing: {path}")
        path.unlink()
        raise
    extractions = [
        CachedExtraction(
            extraction_class=e["extraction_class"],
            extraction_text=e["extraction_text"],
            attributes=e.get("attributes", {}),

        )
        for e in data["extractions"]
    ]
    print(f"[NER] Loaded {len(extractions)} entities from cache: {path}")
    return CachedResult(extractions=extractions, text_preview=data.get("text_preview", ""))
