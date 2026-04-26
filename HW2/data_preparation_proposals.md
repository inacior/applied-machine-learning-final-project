# Data Preparation Proposals — Hybrid NER–SLM Pipeline for Entity-Aware Narrative QA

> Homework 2 — INE410154 Applied Machine Learning — Group: Renan

## 1. External Dataset Options (from Kaggle, HuggingFace, ACL)

The assignment requires using a dataset related to the research project. Below are the best-matching external datasets for literary NER and narrative QA. The Romeo & Juliet dataset is our primary data; any of these can be used in parallel for comparison, training, or benchmarking.

### 1.1 Literary NER Datasets

| Dataset | Source | Size | Entity Types | License | Best For |
|---|---|---|---|---|---|
| **LitBank** | [GitHub](https://github.com/dbamman/litbank) / [HuggingFace](https://huggingface.co/datasets/coref-data/litbank_raw) | 100 novels, 210K tokens | PER, LOC, FAC, GPE, ORG, VEH (ACE 2005 style, nested) | CC-BY 4.0 | Training/fine-tuning a literary NER model; closest to our domain |
| **Fiction-NER-750M** | [HuggingFace](https://huggingface.co/datasets/SaladTechnologies/fiction-ner-750m) | 750M tokens from Gutenberg/AO3 | Character, location, etc. | Open | Large-scale pretraining for fiction NER |
| **ACE 2005** | [LDC](https://catalog.ldc.upenn.edu/LDC2006T06) | ~300K words English (news) | PER, ORG, LOC, GPE, FAC, VEH, WEA | LDC license ($) | Baseline comparison: news-trained NER vs. literary |

**Recommendation:** **LitBank** is the best fit — same domain (fiction), same ACE annotation schema, freely available, and Bamman et al. show that training on LitBank gives +20 F1 points over news-trained models on literary texts.

### 1.2 Narrative QA Datasets

| Dataset | Source | Size | QA Format | Context Length | Best For |
|---|---|---|---|---|---|
| **LiteraryQA** | [HuggingFace](https://huggingface.co/datasets/sapienzanlp/LiteraryQA) / [GitHub](https://github.com/SapienzaNLP/LiteraryQA) | ~10K QA pairs, literary works only | Free-text answer | Full book (~60K words) | Rigorous QA benchmark; cleaned & validated subset of NarrativeQA |
| **NarrativeQA** | [HuggingFace](https://huggingface.co/datasets/deepmind/narrativeqa_manual) | ~46K QA pairs, 1,567 stories | Free-text answer | Full book summaries | Largest narrative QA resource |
| **BookQA** | [GitHub](https://github.com/stangelid/bookqa-who) | 3,427 QA pairs, 614 books | Character identification (classification) | Full book | Character-only QA; directly comparable to our entity-focused pipeline |
| **NovelQA** | [HuggingFace](https://huggingface.co/datasets/NovelQA/NovelQA) | 89 novels, 2,305 QA pairs | Free-text, with difficulty/type labels | 200K+ tokens (ultra-long) | Stress-testing long-context QA |
| **GANDALF** | [ACL Anthology](https://aclanthology.org/2021.mrqa-1.13/) | 20K questions, 177 books | 10-way multiple choice | Full book (~150K words) | Character description matching |

**Recommendation:** **LiteraryQA** is the best QA benchmark to use alongside our 60-question set — it's pre-cleaned, same domain, and uses LLM-as-judge evaluation which aligns with our SLM approach. **BookQA** is also valuable since it specifically targets character identification.

### 1.3 Shakespeare-Specific Datasets (Kaggle)

| Dataset | Source | Size | Format |
|---|---|---|---|
| **Shakespeare Plays Dataset** | [Kaggle](https://www.kaggle.com/datasets/guslovesmath/shakespeare-plays-dataset) | All 36 plays | CSV with speaker, line, act, scene |
| **Shakespeare Plays (Full Text)** | [Kaggle](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) | Complete works | Raw text by play |
| **Shakespeare Plays — Dialogues & Characters** | [Kaggle (romeo_juliet.csv)](https://www.kaggle.com/datasets/umerhaddii/shakespeare-plays-dialogues?select=romeo_juliet.csv) | 38 plays | CSV with play, speaker, dialogue, act, scene |

**Recommendation:** The **Shakespeare Plays — Dialogues & Characters** dataset is structurally identical to our parsed output (speaker-attributed utterances with act/scene metadata). It can serve as a direct comparison — run the same NER pipeline on another play (e.g., *Hamlet*, *Macbeth*) to validate generalizability.

---

## 2. Primary Dataset (Romeo & Juliet) Summary

| Property | Value |
|---|---|
| Source | *Romeo and Juliet* by William Shakespeare |
| Format | Markdown with TEI-like markup (act/scene headers, speaker tags, line numbers) |
| Size | ~25,000 words across 5 acts, 24 scenes, ~1,500 speech segments |
| Unique speakers | ~35 characters |
| Additional data | 60 QA pairs with difficulty labels (easy/medium/hard), linked to line ranges |
| Target task | Entity-aware QA on open-weight small language models (2B–7B) via OpenRouter |

---

## 3. Identified Data Quality Problems

### 3.1 Markup Artifacts in Raw Text

The `raw_dump.md` file contains formatting artifacts that must be cleaned before any NLP processing:

| Problem | Example | Impact |
|---|---|---|
| Line-number anchors | `{#1.1.1}`, `{#1.1.96}` | Pollutes tokenization; inflates vocabulary with meaningless tokens |
| Speaker metadata tags | `[**SAMPSON**]{#speech1}` | Speaker is entangled with markup; cannot be parsed as structured field |
| Blockquote syntax | `> [Gregory, o\' my word...]` | Adds noise to sentence boundaries |
| Stage directions inline | `*Enter ABRAHAM and BALTHASAR*` | Mixed with speech text; needs separate parsing |
| Escape sequences | `o\'`, `\'Tis`, `Ne\'er` | Produces inconsistent tokens; breaks lemmatization |
| HTML-style anchors | `<a id="speech1"/>` (if present) | Purely structural; zero semantic value |

### 3.2 Structural Ambiguities

| Problem | Description |
|---|---|
| Implicit speaker changes | Some speeches lack explicit speaker tags within quick back-and-forth exchanges |
| Anaphoric references | "he", "she", "thee", "thy" → pronouns that need coreference resolution to link to the correct character for QA |
| Nested stage directions | Some directions span multiple lines and are incorrectly segmented |
| Abbreviated character names | `[**First Citizen**]`, `[**Second Servant**]` — inconsistent naming conventions |

### 3.3 Entity-Level Challenges

| Problem | Example |
|---|---|
| Character aliases | Romeo is also "son of Montague", "my cousin", "villain Romeo" |
| Implicit relationships | "My only love sprung from my only hate!" (Juliet about Romeo) — relationship not explicit |
| Family/house affiliations | Characters belong to Capulet vs. Montague but this is not tagged |
| Non-character entities | References to locations (Verona, Mantua, Friar Laurence's cell), events (feast, banishment) |

---

## 4. Common Data Problems in Literary NER and QA — Field Survey & Application

This section maps data problems documented in related work to concrete instances in our Romeo & Juliet dataset, with applied techniques for each.

### 4.1 Domain Shift: News-Trained NER Fails on Literature

**Documented in:** Bamman et al. (2019) — LitBank; Brooke et al. (2016) — LitNER

**What the papers found:**
- NER trained on ACE 2005 (news) achieves 68.8 F1 on news, but drops 23 points to 45.7 F1 on literary text (Bamman et al., Table 2)
- Stanford CoreNLP (news-trained) achieves only 0.751 F-measure on fiction vs. LitNER's 0.792 (Brooke et al., Table 2)
- Literature has proportionally more PER and FAC entities, far fewer ORG and GPE than news (Bamman et al., Figure 1)

**Concrete instance in our data:**
```
Romeo and Juliet has:
- 35+ character entities (PER dominant)  
- ~5 locations (Verona, Mantua, etc.) — few GPE compared to news
- 0 organizations in the ACE sense
- Characters referenced by common nouns: "the Nurse", "the Friar" ← news NER often misses these
```

**Technique applied — Domain-specific NER training:**
```python
# Train on LitBank (literary) instead of CoNLL/ACE (news)
# LitBank has same entity distribution as our data
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"  # baseline news model
)
# Fine-tune on LitBank (available at huggingface.co/datasets/coref-data/litbank_raw)
# Expected gain: +20 F1 points on PER and FAC entities (Bamman et al., Figure 2)
```

---

### 4.2 Multi-Word Names and Title/Honorific Handling

**Documented in:** Vala et al. (2015); Brooke et al. (2016) — LitNER; Bamman et al. (2019) — LitBank

**What the papers found:**
- Off-the-shelf NER often fails to include titles in names: "Mr. Smith" → only "Smith" tagged (Brooke et al., §4)
- ACE 2005 has 47 instances of "Mr." but **zero** mentions of "Mrs." or "Miss" — causes 11.6 point recall gap for female entities (Bamman et al., Table 3)
- Characters frequently referred to by alias/titles: "the Doctor", "Sir", "the Nurse" — not recognized as PER by standard NER (Vala et al.)

**Concrete instances in our data:**

| Problem | Our Data Example | NER Failure |
|---|---|---|
| Title+name | "Lord Capulet", "Lady Montague" | Standard NER often tags only "Capulet"/"Montague" |
| Honorific only | "the Nurse" (named character, no proper name) | Standard NER misses entirely — "nurse" = O |
| Title as name | "Friar Laurence", "Prince Escalus" | "Friar" / "Prince" often dropped from span |
| Female honorific | "Lady Capulet" | 11.6 point recall gap for female PER entities |

**Technique applied — Honorific-aware entity extraction:**
```python
# Prepend honorifics/titles to character name dictionary
HONORIFIC_PATTERNS = {
    'Lord': 'PER', 'Lady': 'PER', 'Friar': 'PER', 'Prince': 'PER',
    'Nurse': 'PER', 'Count': 'PER', 'Countess': 'PER', 'Sir': 'PER',
    'Signior': 'PER', 'County': 'PER',  # County Paris
}

def extract_character_with_title(text: str) -> str | None:
    """Extract full character name including honorific."""
    for title in HONORIFIC_PATTERNS:
        match = re.match(rf'{title}\s+\w+', text)
        if match:
            return match.group(0)  # "Friar Laurence" not just "Laurence"
    return None
```

---

### 4.3 Nested and Non-Named Entity Mentions

**Documented in:** Bamman et al. (2019) — LitBank

**What the papers found:**
- 13.8% of literary entities contain nested structure: "[[the cook]'s sister]" = 3 PER entities
- Entities need not be named: "the boy", "the kitchen", "her daughter" are valid PER/FAC
- Traditional flat NER cannot capture "the elder brother of Isabella's husband"

**Concrete instances in our data:**

| Type | Our Data Example | Nested Structure |
|---|---|---|
| Nested PER | "the only son of your great enemy" (referring to Romeo) | Outer: PER(Romeo) / Inner: PER(Montague) |
| Non-named PER | "my cousin" (Benvolio, referring to Romeo) | "cousin" = PER but no proper name |
| Non-named LOC | "the orchard wall", "the abbey wall" | Descriptive locations |
| Pronoun reference | "he", "she", "thee", "thy" | Coreference needed to resolve to named entity |

**Technique applied — Context window expansion for pronoun resolution:**
```python
def resolve_pronoun(pronoun: str, utterance_index: int, utterances: list[dict]) -> str | None:
    """Resolve 'he'/'she' to the last named speaker in context."""
    for i in range(utterance_index - 1, max(-1, utterance_index - 10), -1):
        prev = utterances[i]
        if prev.get('speaker') and not prev.get('is_stage_direction'):
            return prev['speaker']
    return None

# Example: "he" in Act I Scene I line 116 → resolves to "ROMEO"
```

---

### 4.4 Figurative Language and Metonymy in Literary Text

**Documented in:** Bamman et al. (2019) — LitBank, §3.2

**What the papers found:**
- Metaphor: "the young man was not really a poet; but surely he was a poem" — only "a poet" tagged as PER
- Personification: Animals/objects with dialogue → tagged as PER
- Metonymy: "the kitchen" → tagged as PER when referring to cooks (evoked entity), not FAC (literal)
- "the outraged sentiment of the kitchen was avenged" → "kitchen" = PER (set of cooks)

**Concrete instances in our data:**

| Type | Our Data Example | Challenge |
|---|---|---|
| Personification | "Death is my son-in-law" (Capulet, Act IV) | "Death" = figurative PER, not literal |
| Metonymy | "My house and welcome on their pleasure stay" (Capulet) | "house" = household/family (PER), not FAC |
| Metaphor | "It is the east, and Juliet is the sun" | "sun" ≠ LOC entity despite being a celestial body |

**Technique applied — Context-dependent entity disambiguation:**
```python
# Rule: if a FAC term appears with verbs of emotion/agency, treat as PER (metonymy)
METONYMY_PATTERNS = {
    'house': ['say', 'welcome', 'bid', 'invite', 'pleasure'],
    'kitchen': ['cook', 'prepare', 'dinner', 'outraged'],
}

def disambiguate_metonymy(word: str, context_verbs: list[str]) -> str:
    if word in METONYMY_PATTERNS:
        if any(v in context_verbs for v in METONYMY_PATTERNS[word]):
            return 'PER'  # metonymic use
        return 'FAC'      # literal use
    return 'O'
```

---

### 4.5 Quotation Attribution and Implicit Speaker Ambiguity

**Documented in:** Michel et al. (2024) — Quotation Attribution with Character Embeddings; BookNLP pipeline

**What the papers found:**
- Anaphoric quotes: "he said" → must resolve "he" to character via coreference
- Implicit quotes: dialogue without explicit speaker attribution — hardest case
- BookNLP's 4-step pipeline: NER → name clustering → coreference → quote-to-mention linking
- Character embeddings (stylometric UAR) improve attribution for anaphoric/implicit quotes

**Concrete instances in our data:**

| Type | Our Data Example | Ambiguity |
|---|---|---|
| Explicit speaker | `[**ROMEO**]` `> [Gregory, o' my word...]` | Speaker explicit — easy |
| Implicit speaker | Quick back-and-forth exchanges without speaker tags between lines | Who said "No, marry; I fear thee!" — Gregory? Sampson? |
| Pronoun reference | "Speak, nephew, were you by when it began?" (Montague to Benvolio) | "you" context-dependent |
| Stage direction | `*Enter CAPULET in his gown, and LADY CAPULET*` | Entities appear but don't speak yet |

**Technique applied — Speaker persistence for implicit attribution:**
```python
# After a speaker is established, persist until next explicit speaker change
def attribute_implicit_quotes(utterances: list[dict]) -> list[dict]:
    current_speaker = None
    for u in utterances:
        if u.get('speaker'):
            current_speaker = u['speaker']
        elif not u.get('is_stage_direction'):
            u['speaker'] = current_speaker  # inherit from context
    return utterances
```

---

### 4.6 Genre-Specific Structural Interference

**Documented in:** Brooke et al. (2016) — LitNER (§3.1); DramaAnalysis pipeline (quadrama.github.io)

**What the papers found:**
- Plays have "very different properties in terms of the distribution of names" vs. novels (Brooke et al., §3.1)
- Drama texts have act/scene boundaries, stage directions, speaker tags, and utterances — all intermixed
- BERT fine-tuned on GerDraCor can predict structural elements (act/scene/speaker/direction/utterance) as a 5-way classification task (DramaNLP)

**Concrete instances in our data:**
```
Mixed structural elements in raw_dump.md:
  ### ACT I           ← structural header
  ### SCENE I          ← structural header  
  > *Enter SAMPSON... ← stage direction
  [**SAMPSON**]        ← speaker tag
  > [Gregory, ...]     ← utterance (dialogue)
  > *They fight*       ← stage direction inline
```

**Technique applied — Structural classification for clean parsing:**
```python
def classify_line(line: str) -> str:
    """5-way classification: act, scene, speaker, direction, utterance."""
    if line.startswith('### ACT'):   return 'act_boundary'
    if line.startswith('### SCENE'): return 'scene_boundary'
    if line.startswith('[**') and '**]' in line: return 'speaker_tag'
    if line.strip().startswith('*') and line.strip().endswith('*'): return 'stage_direction'
    if line.startswith('> ['): return 'utterance'
    return 'unknown'
```

---

### 4.7 Name Ambiguity and One-Sense-Per-Text Resolution

**Documented in:** Brooke et al. (2016) — LitNER; Gale et al. (1992)

**What the papers found:**
- "Florence" can be both a city (LOC) and a person name (PER) — but not within the same text
- LitNER assumes **one-sense-per-document**: a name is classified for the entire text based on all its occurrences
- This assumption works for fiction (5.9 occurrences per distinct name per text) but not for short texts (1.6 in Gigaword)

**Concrete instance in our data:**

| Name | Possible Ambiguity | Resolution |
|---|---|---|
| "Paris" | Count Paris (PER) vs. Paris, France (GPE) | Only appears as character name → PER |
| "Verona" | City (GPE) | Only appears as city → GPE |
| "Mantua" | City (GPE) | Only used as location → GPE |

**Technique applied — Text-level entity disambiguation:**
```python
def text_level_classify(name: str, all_contexts: list[str]) -> str:
    """Classify entity type based on all contexts in the play."""
    per_indicators = ['said', 'speak', 'lord', 'count', 'cousin', 'husband', 'wife']
    loc_indicators = ['in', 'from', 'to', 'city', 'street', 'wall', 'gate']

    per_score = sum(1 for ctx in all_contexts if any(w in ctx.lower() for w in per_indicators))
    loc_score = sum(1 for ctx in all_contexts if any(w in ctx.lower() for w in loc_indicators))

    if per_score > loc_score: return 'PER'
    if loc_score > per_score: return 'LOC'
    return 'AMBIGUOUS'
# "Paris" across the play → per_score high (said, lord, count...) → PER
```

---

### 4.8 Summary: Problem-to-Technique Mapping

| # | Problem (from related work) | Occurrence in Our Data | Technique Applied |
|---|---|---|---|
| 1 | Domain shift (news→fiction) | 35+ PER, few ORG/GPE | Fine-tune NER on LitBank |
| 2 | Title/honorific missed | "Lady Capulet", "Friar Laurence" | Honorific-aware extraction regex |
| 3 | Nested/non-named entities | "son of your enemy", "my cousin" | Coreference resolution + span merging |
| 4 | Figurative language | "Juliet is the sun", "Death is my son-in-law" | Context-dependent disambiguation rules |
| 5 | Quotation attribution | Implicit speakers in exchanges | Speaker persistence heuristic |
| 6 | Genre structure mixing | Acts, scenes, directions, tags interleaved | 5-way structural classifier |
| 7 | Name ambiguity | "Paris" (PER vs GPE) | Text-level one-sense-per-document |
| 8 | Gender bias in NER | Female honorifics (Lady, Nurse) | LitBank training (eliminates gender gap) |
| 9 | Segmentation errors | Multi-word names split incorrectly | Brown clustering + phrase improvement (LitNER approach) |

---

## 5. Proposed Preprocessing Pipeline

The pipeline has three stages: **cleaning → structuring → feature engineering**.

### Stage 1: Text Cleaning (Normalization)

Remove all markup artifacts while preserving the underlying text structure.

**Technique 1a: Regex-based markup stripping**

```python
import re

def clean_line(line: str) -> tuple[str, str | None, bool]:
    """Returns (clean_text, speaker, is_stage_direction)"""
    # Detect and extract speaker tags
    speaker_match = re.match(r'^\[\*\*(.+?)\*\*\]\{#speech\d+\}', line)
    speaker = speaker_match.group(1) if speaker_match else None

    # Remove line-number anchors
    clean = re.sub(r'\{#\d+\.\d+\.\d+\}', '', line)

    # Remove blockquote markers
    clean = re.sub(r'^> ', '', clean)

    # Normalize escaped characters (Shakespearean contractions)
    clean = clean.replace("\\'", "'")

    # Detect stage directions
    is_stage = bool(re.match(r'^\*.*\*$', clean.strip())) or line.startswith('*')

    return clean.strip(), speaker, is_stage
```

**Example — before and after:**

```
Before:  > [Gregory, o\' my word, we\'ll not carry coals.]{#1.1.1}
After:   Gregory, o' my word, we'll not carry coals.

Before:  [**SAMPSON**]{#speech1}
After:   (extracted speaker="SAMPSON", clean_text="")
```

**Technique 1b: Unicode normalization & whitespace consolidation**

```python
import unicodedata

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)     # normalize paragraph breaks
    return text.strip()
```

**Why this matters:** Unnormalized text inflates tokenizer vocabulary, degrades embedding quality, and causes LLMs to produce artifacts. For small models (2B–7B), every token counts — cleaner input = less waste on noise.

---

### Stage 2: Structural Parsing (Drama-Specific Structuring)

Convert the cleaned text into structured data: utterances with metadata.

**Technique 2a: Utterance extraction into structured records**

```python
import csv

def extract_utterances(text: str) -> list[dict]:
    utterances = []
    current_speaker = None
    current_scene = None
    current_act = None

    for line in text.split('\n'):
        # Detect act/scene headers
        if '### ACT' in line:
            current_act = line.strip()
        elif '### SCENE' in line:
            current_scene = line.strip()

        speaker = extract_speaker(line)
        is_stage = is_stage_direction(line)
        is_speech = not is_stage and speaker is None

        if speaker:
            current_speaker = speaker

        if current_speaker and is_speech:
            utterances.append({
                'act': current_act,
                'scene': current_scene,
                'speaker': current_speaker,
                'text': clean_line(line),
                'is_stage_direction': False
            })
        elif is_stage:
            utterances.append({
                'act': current_act,
                'scene': current_scene,
                'speaker': None,
                'text': clean_line(line),
                'is_stage_direction': True
            })

    return utterances
```

**Example output row:**

| act | scene | speaker | text |
|---|---|---|---|
| ACT I | SCENE I | SAMPSON | Gregory, o' my word, we'll not carry coals. |
| ACT I | SCENE I | None | *Enter ABRAHAM and BALTHASAR* |

**Technique 2b: Scene-level speaker co-occurrence graph construction**

From the structured utterances, build an adjacency matrix of which characters appear together in scenes — this is critical for the entity graph in the downstream QA pipeline.

```python
from collections import defaultdict
from itertools import combinations

def build_co_occurrence_graph(utterances):
    scene_speakers = defaultdict(set)
    for u in utterances:
        if u['speaker'] and not u['is_stage_direction']:
            key = (u['act'], u['scene'])
            scene_speakers[key].add(u['speaker'])

    edges = defaultdict(int)
    for speakers in scene_speakers.values():
        for s1, s2 in combinations(speakers, 2):
            edge = tuple(sorted([s1, s2]))
            edges[edge] += 1
    return edges
```

**Why this matters:** Co-occurrence is a proxy for relationships. A speaker graph allows the QA system to answer "Who was present when X happened?" without reading the full text.

---

### Stage 3: Feature Engineering

Build features that augment the raw text before passing it to the small language model.

#### 3.1 Entity Profile Feature

For each character, create a concise profile snippet with key facts extracted deterministically:

```python
def build_character_profile(name: str, utterances: list[dict]) -> str:
    lines = [u['text'] for u in utterances if u['speaker'] == name]
    scenes = set((u['act'], u['scene']) for u in utterances if u['speaker'] == name)
    first_appearance = min(scenes, key=lambda s: int(s[0][-1]) * 100 + int(s[1][-1]))
    num_scenes = len(scenes)

    # Extract relationships from co-occurrence
    related_to = get_related_characters(name, co_occurrence_graph)

    return f"""[CHARACTER: {name}]
House: {'Capulet' if name in CAPULET_HOUSE else 'Montague' if name in MONTAGUE_HOUSE else 'Neutral'}
First appears: {first_appearance}
Appears in {num_scenes} scene(s)
Relationships: {', '.join(related_to)}"""
```

**Example:**

```
[CHARACTER: ROMEO]
House: Montague
First appears: ACT I, SCENE I
Appears in 14 scene(s)
Relationships: Benvolio (cousin), Mercutio (friend), Juliet (wife), Friar Laurence (confessor), Montague (father), Lady Montague (mother), Tybalt (enemy, killed by), Paris (rival)
```

#### 3.2 Conversation Context Feature

For each utterance, prepend information about who else is present in the scene:

```python
def build_conversation_context(utterance: dict, all_utterances: list[dict]) -> str:
    scene_speakers = [u['speaker'] for u in all_utterances
                      if u['act'] == utterance['act']
                      and u['scene'] == utterance['scene']
                      and u['speaker'] is not None]
    unique_speakers = list(dict.fromkeys(scene_speakers))  # preserve order
    return f"[Scene: {utterance['scene']} | Present: {', '.join(unique_speakers)}]"
```

**Example:**

```
[Scene: SCENE V. A hall in Capulet's house. | Present: First Servant, Second Servant, CAPULET, Second Capulet, ROMEO, TYBALT, JULIET, Nurse, BENVOLIO]
```

#### 3.3 Entity Augmentation for QA

When querying the SLM with a question, prepend the structured entity context:

```
## Context
[Scene: SCENE V ... | Present: ROMEO, TYBALT, CAPULET, JULIET, ...]

[CHARACTER: ROMEO]
House: Montague | Relationships: Juliet (wife), Tybalt (enemy, killed), ...

[CHARACTER: TYBALT]
House: Capulet | Relationships: Juliet (cousin), Romeo (enemy, killed by), ...

## Question
In Scene V, by what identifying signal does Tybalt detect Romeo at the feast, and whose intervention prevents an immediate assault?

## Source text
[cleaned scene text]
```

**Why this works:** Small models (2B–7B) excel when contextual information is explicitly present rather than requiring inference across long passages. The entity profile acts as a "cheat sheet" that anchors the model's attention to relevant entities before it processes the full text.

#### 3.4 Difficulty-Based Context Windows

Use the difficulty labels from the QA benchmark to vary context window size:

```python
def get_context_window(question_difficulty: str) -> str:
    windows = {
        'easy': 'current_scene',          # 1 scene (~200 words)
        'medium': 'adjacent_scenes',       # 3 scenes (~600 words)
        'hard': 'full_act',                # Full act (~2500 words)
    }
    return windows[question_difficulty]
```

**Why this matters:** This tests whether the structured entity context enables shorter (more token-efficient) context windows, which directly impacts API costs via OpenRouter.

---

## 6. Visualization & Exploration Ideas

### 6.1 Speaker Distribution & Frequency

Plot how many lines each character has — reveals narrative centrality.

```
ROMEO: ████████████████████████████████████████ (148 speeches)
JULIET: ██████████████████████████████ (118 speeches)
Nurse: ████████████████████ (90 speeches)
FRIAR LAURENCE: ██████████████ (51 speeches)
...
```

### 6.2 Scene-Level Interaction Heatmap

Which character pairs share the most scenes?

```
            ROM JUL NUR CAP TYB MER BEN FRI PAR
ROMEO         -   7   3   4   6   5  11   5   1
JULIET        7   -   6  10   3   0   2   2   2
NURSE         3   6   -   4   0   0   0   0   0
...
```

### 6.3 Act-by-Act Entity Density

NER entity mentions per act — shows where the plot is character-rich vs. action-heavy.

### 6.4 Question Difficulty Distribution

```python
easy:    20 questions (33%) — mostly identity lookup
medium:  25 questions (42%) — relationship, causality
hard:    15 questions (25%) — multi-hop inference
```

---

## 7. Data Augmentation (Optional Enhancement)

### 7.1 Domain Adaptation via Name Substitution

For improving NER performance on literary texts, we can augment the training data:

**Technique:**

> From [Bert meets d'Artagnan, Dufour et al. 2022](https://hal.univ-lorraine.fr/EC-NANTES/hal-03617722v1):
> Replace modern entity names in CoNLL-2003 with literary-style character names from the target domain corpus, then fine-tune a BERT-based NER model on the augmented set.

```python
# Pseudocode
conll_names = ["John", "Mary", "London"]
literary_names = ["Romeo", "Juliet", "Verona"]

for sentence in conll_corpus:
    for conll_name, lit_name in zip(conll_names, literary_names):
        sentence = sentence.replace(conll_name, lit_name)
```

This yields a model that better recognizes literary character names and locations, reducing false negatives in the downstream NER pass.

### 7.2 Preprocessing + Augmentation Pipeline for QA Benchmark

> From [Duong & Nguyen-Thi (2021)](https://doi.org/10.1186/s40649-020-00080-x): Preprocessing before augmentation consistently outperforms either technique alone — clean data yields higher-quality synthetic examples.

Our QA benchmark has only 60 questions, which is a very small dataset. The paper shows that combining preprocessing with data augmentation significantly boosts classifier accuracy when training data is limited.

#### 7.2.1 EDA for Question Augmentation

Easy Data Augmentation (EDA) from Wei & Zou (2019), validated by Duong & Nguyen-Thi, generates synthetic QA pairs from existing ones:

```python
import random

def synonym_replace(question: str, n: int = 1) -> str:
    """Replace n non-stop words with synonyms (via WordNet)."""
    words = question.split()
    candidates = [w for w in words if w.lower() not in STOP_WORDS]
    for _ in range(min(n, len(candidates))):
        word = random.choice(candidates)
        synonyms = get_synonyms(word)
        if synonyms:
            words[words.index(word)] = random.choice(synonyms)
    return ' '.join(words)

def random_swap(question: str, n: int = 1) -> str:
    """Randomly swap two non-stop words n times."""
    words = question.split()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def random_delete(question: str, p: float = 0.1) -> str:
    """Randomly delete each word with probability p."""
    words = [w for w in question.split() if random.random() > p]
    return ' '.join(words) if words else question
```

**Example — augmenting a QA pair:**

```
Original: "Which gesture does Sampson intentionally direct at the Montague servants?"

Synonym Replace: "Which action does Sampson intentionally direct at the Montague servants?"
Random Swap:     "Which gesture Sampson does intentionally direct at the Montague servants?"
Random Delete:   "Which gesture Sampson intentionally at the Montague servants?"
```

**Why this matters:** A single QA pair can generate 4–5 variants that test the same knowledge with different phrasing. This evaluates whether the SLM truly understands the question or is pattern-matching on surface forms. With 60 original → ~240 augmented questions, we can measure robustness.

#### 7.2.2 Back Translation for Question Paraphrasing

Translate questions to an intermediate language (e.g., Portuguese — since this is a Brazilian university project) and back to English to get natural paraphrases:

```python
from deep_translator import GoogleTranslator

questions_portuguese = [
    "Qual gesto Sampson dirige intencionalmente aos servos Montague?",
    "Que negação ele usa imediatamente para limitar exposição legal?",
]
back_to_english = GoogleTranslator(source='pt', target='en').translate_batch(questions_portuguese)
# "What gesture does Sampson intentionally direct at the Montague servants?"
# "What denial does he immediately use to limit legal exposure?"
```

**Benefit:** Tests whether the SLM can answer the same knowledge question when phrased with different vocabulary, measuring generalization rather than memorization. Duong & Nguyen-Thi found that preprocessing before back-translation yields higher quality paraphrases.

#### 7.2.3 Preprocessing Hierarchy

The paper establishes a clear order of operations that we adopt:

```
1. Clean (markup removal, normalization)   ← Stage 1
2. Normalize (lowercase, stop words [optional for QA], unicode)   ← Stage 1
3. Handle special tokens (negation markers, character names)   ← Stage 2/3
4. Augment (EDA / back translation)   ← Section 5.2
5. Vectorize (tokenize for SLM input)
6. Evaluate (raw vs. augmented comparison)
```

The key finding: running augmentation on preprocessed data (steps 1–3 first) produces **significantly better results** than augmenting raw data.

---

## 8. Implementation Roadmap

| Step | Description | Output |
|---|---|---|---|
| 1 | Clean markup & normalize text | Clean `.txt` file |
| 2 | Parse into structured CSV | `utterances.csv` with speaker, scene, act columns |
| 3 | Build co-occurrence graph | JSON graph of character relationships |
| 4 | Extract entity profiles | Per-character metadata with relationships |
| 5 | Generate augmented QA prompts | 60 original + ~240 augmented prompts (raw vs. entity-augmented) |
| 6 | Exploratory visualizations | Distribution plots, heatmaps, entity density |
| 7 | Run NER on LitBank comparison | NER F1 scores on LitBank vs. Romeo & Juliet |
| 8 | Run SLM benchmarking via OpenRouter | Accuracy comparison raw vs. augmented, plus LiteraryQA comparison |

---

## 9. References

- Bamman, D. et al. (2019). *An Annotated Dataset of Literary Entities.* NAACL. https://aclanthology.org/N19-1220/
- Brooke, J. et al. (2016). *Bootstrapped Text-level Named Entity Recognition for Literature.* ACL. https://aclanthology.org/P16-2056/
- Vala, H. et al. (2015). *Mr. Bennet, his coachman, and the Archbishop walk into a bar but only one of them gets recognized: On the difficulty of detecting characters in literary texts.* EMNLP. https://aclanthology.org/D15-1088/
- Gale, W. et al. (1992). *One sense per discourse.* DARPA Speech and Natural Language Workshop. https://aclanthology.org/H92-1045/
- Dufour, R. et al. (2022). *BERT meets d'Artagnan: Data Augmentation for Robust Character Detection in Novels.* LREC. https://hal.univ-lorraine.fr/EC-NANTES/hal-03617722v1
- Michel, G. et al. (2024). *Improving Quotation Attribution with Fictional Character Embeddings.* EMNLP Findings. https://aclanthology.org/2024.findings-emnlp.744/
- Yang, F. et al. (2022). *Character Identification in Literary Texts.* AAAI. https://cdn.aaai.org/ojs/21709/21709-13-25722-1-2-20220628.pdf
- Duong, H.T. & Nguyen-Thi, T.A. (2021). *A review: preprocessing techniques and data augmentation for sentiment analysis.* Computational Social Networks 8, 1. https://doi.org/10.1186/s40649-020-00080-x
- Wei, J. & Zou, K. (2019). *EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks.* ICLR. https://arxiv.org/abs/1901.11196
- Angelidis, S. et al. (2019). *Book QA: Stories of Challenges and Opportunities.* MRQA Workshop. https://aclanthology.org/D19-5819/
- Bonomo, T. et al. (2025). *LiteraryQA: Towards Effective Evaluation of Long-document Narrative QA.* EMNLP. https://aclanthology.org/2025.emnlp-main.1729/
- Salad Technologies. (2025). *Fiction-NER-750M.* HuggingFace. https://huggingface.co/datasets/SaladTechnologies/fiction-ner-750m
- Group 5 (2024.1). *Data Preparation — INE410154 AML.* (reference pipeline: standardization, data augmentation, feature selection, filtering)
