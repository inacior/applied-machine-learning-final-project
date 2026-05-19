"""spaCy Graph — spaCy entity extraction + graph builder.

Runs locally with spaCy (no API calls). Extracts characters, families, and locations
from the dialogue column, builds co-occurrence relationship graphs, and formats a
structured overview for LLM prompts.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from bench.config import HW3_DIR

SPACY_GRAPH_CACHE_DIR = HW3_DIR / ".ner_cache" / "spacy_graph"


def _get_nlp():
    """Lazy-load spaCy English model with helpful error on missing download."""
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required for spacy_graph method. "
            "Install it:  pip install spacy>=3.7.0\n"
            "Then download model:  python -m spacy download en_core_web_sm"
        ) from exc

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise OSError(
            "spaCy model 'en_core_web_sm' not found. "
            "Download it with:  python -m spacy download en_core_web_sm"
        ) from exc


def _extract_entities_from_dialogue(dialogue: str, nlp) -> dict[str, list[str]]:
    """Run spaCy NER on a single dialogue string."""
    doc = nlp(dialogue)
    entities: dict[str, list[str]] = {
        "characters": [],
        "families": [],
        "locations": [],
    }
    for ent in doc.ents:
        text = ent.text.strip()
        if not text:
            continue
        if ent.label_ == "PERSON":
            entities["characters"].append(text)
        elif ent.label_ == "ORG":
            entities["families"].append(text)
        elif ent.label_ == "GPE":
            entities["locations"].append(text)
    return entities


def _build_profiles_and_graph(
    dataset_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], int]]:
    """Build entity profiles and co-occurrence graph from dataset."""
    nlp = _get_nlp()

    row_entities = [
        _extract_entities_from_dialogue(d, nlp)
        for d in dataset_df["dialogue"].astype(str)
    ]

    profiles: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"mentions": 0, "relationships": defaultdict(int)}
    )
    edges: dict[tuple[str, str], int] = defaultdict(int)

    for entities in row_entities:
        chars = list(dict.fromkeys(entities["characters"]))
        for char in chars:
            profiles[char]["mentions"] += 1
            for other in chars:
                if other != char:
                    profiles[char]["relationships"][other] += 1

        seen_pairs = set()
        for i, c1 in enumerate(chars):
            for c2 in chars[i + 1 :]:
                pair = tuple(sorted((c1, c2)))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges[pair] += 1

    clean_profiles = {
        name: {
            "mentions": data["mentions"],
            "relationships": dict(data["relationships"]),
        }
        for name, data in profiles.items()
    }

    return clean_profiles, dict(edges)


def _format_overview(
    profiles: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str], int],
    top_n_chars: int | None = None,
    top_n_edges: int | None = None,
) -> str:
    lines: list[str] = []

    sorted_chars = sorted(
        profiles.items(), key=lambda x: x[1]["mentions"], reverse=True
    )
    char_slice = sorted_chars if top_n_chars is None else sorted_chars[:top_n_chars]

    lines.append("## Character Profiles")
    lines.append("")
    for name, data in char_slice:
        mentions = data["mentions"]
        rels = data.get("relationships", {})
        sorted_rels = sorted(rels.items(), key=lambda x: x[1], reverse=True)
        rel_str = ", ".join(f"{r} ({c})" for r, c in sorted_rels) if sorted_rels else "none"
        lines.append(f"- **{name}**: {mentions} mentions. Related to: {rel_str}")

    if top_n_chars is not None and len(sorted_chars) > top_n_chars:
        lines.append(f"- ... and {len(sorted_chars) - top_n_chars} more characters")

    lines.append("")

    sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
    edge_slice = sorted_edges if top_n_edges is None else sorted_edges[:top_n_edges]
    if edge_slice:
        lines.append("## Key Relationships (co-occurrence count)")
        lines.append("")
        for (a, b), weight in edge_slice:
            lines.append(f"- {a} ↔ {b}: {weight}")
        if top_n_edges is not None and len(sorted_edges) > top_n_edges:
            lines.append(
                f"- ... and {len(sorted_edges) - top_n_edges} more relationships"
            )
        lines.append("")

    return "\n".join(lines)


def _cache_path_for_dataset(dataset_path: str | Path) -> Path:
    """Deterministic cache path based on dataset file name."""
    name = Path(dataset_path).stem
    SPACY_GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SPACY_GRAPH_CACHE_DIR / f"{name}_spacy_graph.json"


def _save_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_cache(cache_path: Path) -> dict | None:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return None


def get_spacy_graph_context(
    dataset_path: str | Path,
    dataset_df: pd.DataFrame | None = None,
    cache_path: Path | None = None,
    top_n_chars: int | None = None,
    top_n_edges: int | None = None,
) -> str:
    if cache_path is None:
        cache_path = _cache_path_for_dataset(dataset_path)

    cached = _load_cache(cache_path)
    if cached is not None:
        return _format_overview(
            cached["profiles"],
            {tuple(k.split("::")): v for k, v in cached["edges"].items()},
            top_n_chars=top_n_chars,
            top_n_edges=top_n_edges,
        )

    if dataset_df is None:
        dataset_df = pd.read_csv(dataset_path)

    print(f"[spacy_graph] Running spaCy NER on {len(dataset_df)} rows...")
    start = time.time()
    profiles, edges = _build_profiles_and_graph(dataset_df)
    elapsed = time.time() - start
    print(
        f"[spacy_graph] Extracted {len(profiles)} characters, "
        f"{len(edges)} relationships in {elapsed:.1f}s"
    )

    _save_cache(
        cache_path,
        {
            "profiles": profiles,
            "edges": {f"{a}::{b}": w for (a, b), w in edges.items()},
        },
    )

    return _format_overview(profiles, edges, top_n_chars=top_n_chars, top_n_edges=top_n_edges)


def export_graph_visualization(
    dataset_path: str | Path,
    output_dir: str | Path = "ner_results",
) -> None:
    """Optional: export GEXF and interactive HTML visualizations.

    Not used during benchmarking — useful for reports.
    """
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError(
            "networkx and pyvis are required for visualization export. "
            "Install them:  pip install networkx pyvis"
        ) from exc

    dataset_df = pd.read_csv(dataset_path)
    profiles, edges = _build_profiles_and_graph(dataset_df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    profiles_path = out / "entity_profiles.json"
    profiles_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False))
    print(f"Saved: {profiles_path}")

    G = nx.Graph()
    for (a, b), weight in edges.items():
        G.add_edge(a, b, weight=weight)

    gexf_path = out / "entity_relationship_graph.gexf"
    nx.write_gexf(G, gexf_path)
    print(f"Saved: {gexf_path}")

    net = Network(notebook=False, height="800px", width="100%", bgcolor="#ffffff")
    for node in G.nodes:
        net.add_node(node, label=node, title=f"{node}")
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=data["weight"], title=f"weight: {data['weight']}")
    net.repulsion(node_distance=200, central_gravity=0.3)
    html_path = out / "entity_relationship_graph.html"
    net.write_html(str(html_path))
    print(f"Saved: {html_path}")
