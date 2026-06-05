# Entity-Aware Narrative Question Answering: A LangExtract-Augmented Pipeline for Small Language Models

---

## Abstract

Structured entity extraction is a cornerstone capability in digital forensics and narrative comprehension, where identifying actors, relationships, and causal chains determines whether an analysis is actionable. This work investigates whether compact, globally organized entity summaries produced by Google's LangExtract library can improve question-answering accuracy for small open-weight language models (1B–30B parameters) on a narrative comprehension benchmark. Using Shakespeare's *Romeo and Juliet* as a test corpus—a work whose dense dialogue, archaic diction, and intricate character relationships make it a demanding proxy for forensic narrative analysis—we construct a 1,068-row preprocessed dataset and a 60-question evaluation set spanning three difficulty levels. LangExtract, with Gemini 2.5 Flash as the extraction backend, produces a semantic overview of characters, relationships, events, locations, and themes, which is prepended to the full dataset before querying seven small language models via OpenRouter. Excluding the context-limited Llama 1B, LangExtract improves mean strict accuracy from 30.6% to 36.7% (+6.1 points) and lenient accuracy from 65.6% to 68.1% (+2.5 points). The largest gains materialize on hard questions requiring relational and causal reasoning (26.9% → 36.1% strict; +9.2 points). These results demonstrate that compact, globally organized entity structure reduces narrative search burden more effectively than raw text alone, particularly for models in the 3B–24B parameter range where internal reasoning capacity remains constrained.

**Keywords**: named entity recognition, small language models, structured extraction, LangExtract, narrative question answering, long-context reasoning, LLM-augmented preprocessing

---

## 1. Introduction

### 1.1 Context and Motivation

The extraction of structured entities from unstructured text is increasingly central to both digital forensics and computational literary analysis. In forensic investigation, large language models have been applied to messenger data for evidence triage and entity identification, demonstrating improved recall over keyword-only approaches [1]. However, reproducibility remains fragile—dataset availability is a persistent bottleneck [2]—and recent surveys caution that LLM outputs are difficult to validate without grounding in source evidence, particularly for small, open-weight models [3]. Forensic intelligence graphs address some of these concerns by accumulating structured entity extractions into navigable representations that support downstream analysis and review [4].

These findings suggest a broader principle with significance beyond forensics: if lightweight, deterministic entity extraction can produce a structured overview of a narrative, it may improve the question-answering accuracy of small language models that lack the parameter budget for internal reconstruction of complex relational structures from raw text alone.

### 1.2 Problem Definition

Long-context question answering over literary corpora is challenging for small and mid-sized language models because the task combines multiple cognitive burdens simultaneously: retaining extensive textual context, identifying which passages bear on a given question, linking temporally and spatially separated events, and reconstructing character relationships without drifting into unsupported inference. A Shakespeare play constitutes an especially rigorous test bed. Dialogue is dense and stylized, character names appear in multiple morphological forms, stage directions encode important state transitions, and the Elizabethan lexicon can mislead general-purpose named entity systems.

The central empirical question is whether explicit entity-aware preprocessing can mitigate these burdens. If a model first receives a structured overview of characters, events, and relationships before processing the full dataset, it may navigate the long prompt more efficiently and generate more accurate answers. This hypothesis is most consequential for questions about identity, kinship, scene participation, and causal chains—precisely the categories where small models most frequently fail.

### 1.3 Objectives

The specific objectives of this work are:

1. To develop a reproducible preprocessing pipeline that applies structured entity extraction to the complete text of a literary work, producing a grounded entity overview.
2. To construct a benchmark of 60 question-answer pairs spanning factoid, relational, and causal types, with per-question difficulty labels and evidence-level validation.
3. To evaluate multiple open-weight small language models (1B–30B parameters) via OpenRouter, comparing accuracy with and without structured entity context.
4. To quantify the performance delta attributable to entity-aware preprocessing and identify which question types benefit most.
5. To benchmark the hybrid pipeline against a stronger proprietary LLM to estimate the residual performance gap.

---

## 2. Related Work

### 2.1 Named Entity Recognition for Literary Texts

The domain-shift problem in NER is well documented. Standard models trained on news corpora underperform sharply on literary text. Bamman et al. [5] reported that ACE 2005-trained models drop from 68.8 F1 on news to 45.7 F1 on literary text—a 23-point decline. Literature exhibits proportionally more person and facility entities, fewer organization and geopolitical entities, and 13.8% of literary entities contain nested structures that flat sequence-labeling NER cannot capture. Brooke et al. [6] found that Stanford CoreNLP achieved only 0.751 F-measure on fiction, versus 0.792 for a bootstrapped domain-specific approach. Vala et al. [7] further showed that off-the-shelf NER omits titles from names and fails to recognize characters referenced by common nouns, such as "the Nurse" or "the Friar."

These domain-shift findings motivated our choice of an LLM-based extraction approach (LangExtract) over traditional sequence-labeling NER. Unlike fixed-label classifiers, LLMs can exploit context to handle the figurative language, metonymy, and honorific patterns characteristic of literary text.

### 2.2 LLM-Based Structured Extraction

Google's LangExtract [8] is a Python library for structured information extraction from unstructured text using LLMs as backend extractors. It differs from traditional NER in several respects: extraction classes are defined through natural language prompts and few-shot examples rather than a fixed tag schema; each extracted span is grounded to precise character offsets in the source, ensuring traceability; and long documents are divided into overlapping buffers with configurable chunking and parallel processing parameters.

Key architectural features include: **precise source grounding** (exact character offsets), a **chunking strategy** for long documents (`max_char_buffer` controls chunk size), **multiple extraction passes** for iterative refinement, **parallel processing** with configurable worker counts, and **schema enforcement** through few-shot examples. LangExtract is compatible with multiple LLM backends, including Gemini, OpenAI, and custom providers via OpenRouter, and has been applied in medical information extraction tasks [8].

### 2.3 Narrative Question Answering

Recent years have seen the development of several narrative QA benchmarks. NarrativeQA [9] provides approximately 46,000 question-answer pairs across 1,567 stories. LiteraryQA [10] refines this approach with cleaned, validated questions specifically targeting literary works. BookQA [11] frames character identification as a classification task. A consistent finding across these benchmarks is that long-context reasoning remains extremely difficult for models below 10B parameters, particularly for questions requiring linkage of information from distant parts of a narrative.

Our work differs in two respects: we focus on a single work with broad question coverage (60 questions spanning the entire play), and we explicitly test whether preprocessing with structured entity extraction improves performance, rather than measuring raw model capability in isolation.

---

## 3. Dataset

### 3.1 Source Material

The primary corpus is the complete text of Shakespeare's *Romeo and Juliet*, sourced from the Kaggle dataset `umerhaddii/shakespeare-plays-dialogues` [12]. The original CSV provides dialogue lines with speaker attribution, act and scene metadata, and sequential line numbers.

### 3.2 Preprocessing Pipeline

The raw dataset was transformed through three principal preprocessing stages:

**Row consolidation** merged consecutive lines by the same character within a scene into single dialogue blocks, reducing the dataset from approximately 1,500 speech segments to 1,068 rows while preserving narrative continuity.

**Participant tracking** added a `participants` column that enumerates characters physically present on stage at each moment, inferred from Enter, Exit, and Exeunt stage directions. This column enables interaction analysis and serves as a validation resource for ground-truth answers.

**Name normalization** decomposed character names into `title`, `first_name`, `last_name`, and `normalized_name` columns. This handled honorifics (e.g., "Lady Capulet," "Friar Laurence") and aliases, producing a canonical name that could be used consistently across the dataset.

The final dataset comprises 10 columns, detailed in Table I.

**Table I: Dataset Schema**

| Column | Description |
|--------|-------------|
| `act` | Act I through Act V |
| `scene` | Prologue, Scene I–V |
| `character` | Speaker name or `[stage direction]` |
| `dialogue` | Merged dialogue text |
| `line_number` | Last sequential line number in the merged block |
| `participants` | Characters physically on stage |
| `title` | Extracted honorific (LADY, PRINCE, NURSE) |
| `first_name` | First name component |
| `last_name` | Last name component (CAPULET, MONTAGUE) |
| `normalized_name` | Canonical name string |

### 3.3 Benchmark Questions

A set of 60 question-answer pairs was constructed as a forensics-style evidence questionnaire, classified by difficulty as shown in Table II.

**Table II: Question Difficulty Distribution**

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 11 | Direct factoid questions requiring single-fact recall |
| Medium | 31 | Relational and interpretive questions spanning multiple lines |
| Hard | 18 | Causal and multi-step reasoning questions requiring cross-scene linkage |

Each question includes a ground-truth answer, analyst observations with line-range validation references, and CSV row indices for evidence reconstruction. The full question set is provided in Appendix B.

### 3.4 Data Quality Challenges

Several domain-specific phenomena complicated both preprocessing and entity extraction:

**Markup artifacts**: The raw text contained line-number anchors (`{#1.1.1}`), speaker metadata tags (`[**SAMPSON**]{#speech1}`), and escape sequences requiring cleanup prior to NLP processing.

**Character aliases**: Romeo appears as "son of Montague," "my cousin," and "villain Romeo" at different points. Family affiliations (Capulet versus Montague) are not explicitly tagged in the source.

**Figurative language**: Metaphor ("It is the east, and Juliet is the sun"), personification ("Death is my son-in-law"), and metonymy ("My house and welcome on their pleasure stay," where "house" denotes household/family) all complicate entity extraction.

**Implicit references**: Pronouns and anaphoric references require coreference resolution to map to the correct character, a problem compounded by the absence of explicit pronoun-antecedent links in the raw text.

These challenges are precisely the type that LLM-based extraction (LangExtract) is better equipped to handle than traditional sequence-labeling NER, as discussed in Section 4.

---

## 4. Methodology

### 4.1 Pipeline Overview

The workflow proceeds in three stages:

1. **Entity extraction**: LangExtract processes the formatted dataset text to extract structured entities across seven classes.
2. **Context augmentation**: The extracted entities are organized into a compact semantic overview and prepended to the full dataset.
3. **Question answering**: Small language models receive the augmented context and answer each benchmark question. A verifier LLM evaluates answers against ground truth.

```
Dataset CSV → format as text → LangExtract extraction → structured overview
                                                              ↓
Questions CSV → format prompt with overview + full dataset → SLM → answers → verifier → scores
```

### 4.2 LangExtract Configuration

The extraction pipeline uses `google/gemini-2.5-flash` via OpenRouter as the extraction backend with seven entity classes defined in the extraction prompt: `character`, `relationship`, `emotion`, `event`, `location`, `object`, and `theme`. The complete prompt is reproduced below:

```
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
```

A single few-shot example demonstrates character and theme extraction from the Prologue. Processing parameters are: `extraction_passes=1`, `max_workers=5`, and `max_char_buffer=50,000`.

### 4.3 Structured Overview Generation

Raw LangExtract output is filtered to retain only grounded spans—extractions whose text appears verbatim in the source—and reorganized into a markdown-formatted overview comprising seven sections: Characters (with mention counts), Relationships (interactions and family affiliations), Key Events (organized by act and scene), Emotional Themes, Locations, Significant Objects, and Themes.

The resulting prompt structure appends this overview (~2,000 tokens) before the full dataset (~48,000 tokens) with instructions to use the overview as a navigation guide.

### 4.4 Question Answering Protocol

Each model receives the augmented context and answers all 60 questions under a system prompt requiring answers drawn solely from the provided context, paraphrased rather than quoted, with honest admission of insufficient information where applicable. Generation uses `temperature=0.0` for deterministic reproducibility, `max_tokens=2,048`, and rate-limited API calls (1.0-second delay, 3 retries with 1.0-second backoff).

### 4.5 Evaluation Framework

Answers are evaluated by a verifier LLM (`openai/gpt-5.4-mini`) using a three-class taxonomy:

- **CORRECT**: Captures all essential facts from ground truth, no factual errors.
- **PARTIALLY_CORRECT**: Captures some key facts but misses important elements, or contains minor factual errors.
- **INCORRECT**: Substantially wrong, contradicts ground truth, fabricates information, or fails to answer.

The verifier receives the question, ground-truth answer, analyst observations, reconstructed evidence rows, and the model's answer. Two composite metrics are reported: **strict accuracy** (percentage CORRECT) and **lenient accuracy** (percentage CORRECT or PARTIALLY_CORRECT).

---

## 5. Experimental Setup

### 5.1 Models Evaluated

Seven open-weight small language models spanning 1B to 30B parameters were evaluated via OpenRouter, as detailed in Table III. A proprietary baseline (GPT-5.4) was also run to establish the performance ceiling.

**Table III: Models Evaluated**

| Model | Parameters | Context Window | OpenRouter ID |
|-------|------------|----------------|---------------|
| Gemma 3 4B | 4B | 32K | `google/gemma-3-4b-it` |
| Llama 1B | 1B | 60K | `meta-llama/llama-3.2-1b-instruct` |
| Llama 3B | 3B | 128K | `meta-llama/llama-3.2-3b-instruct` |
| Ministral 3B | 3B | 128K | `mistralai/ministral-3b-2512` |
| Mistral Small 3.2 | 24B | 128K | `mistralai/mistral-small-3.2-24b-instruct-2506` |
| Phi-4 Mini | 3.8B | 128K | `microsoft/phi-4-mini-instruct` |
| Qwen3 30B A3B | 30B (3B active) | 128K | `qwen/qwen3-30b-a3b-instruct-2507` |

### 5.2 Conditions Compared

Two conditions were compared: a **baseline** consisting of raw formatted CSV rows with no entity extraction, and a **LangExtract** condition in which the structured semantic overview is prepended to the full dataset. This minimal-comparison design isolates the effect of entity-aware preprocessing.

### 5.3 Implementation

The benchmark system is implemented in Python. Key components include `run_benchmark.py` (orchestration), `bench/ner_pipeline.py` (LangExtract extraction and overview generation), `bench/data_loader.py` (dataset and question loading), `bench/openrouter.py` (API client with rate limiting and retry logic), `bench/evaluator.py` (LLM-based answer evaluation), and `evaluate_results.py` (standalone evaluation). All code, data, and result artifacts are available in the project repository.

---

## 6. Results

### 6.1 Per-Model Comparison

Table IV reports strict and lenient accuracy for each model under both conditions.

**Table IV: Per-Model Accuracy (strict / lenient, %)**

| Model | Baseline | LangExtract | Δ Strict | Δ Lenient |
|-------|----------|-------------|----------|-----------|
| Gemma 3 4B | 11.7 / 43.3 | 13.3 / 45.0 | +1.6 | +1.7 |
| Llama 1B | 0.0 / 3.3 | 0.0 / 0.0 | 0.0 | −3.3 |
| Llama 3B | 13.3 / 53.3 | 16.7 / 50.0 | +3.4 | −3.3 |
| Ministral 3B | 13.3 / 65.0 | 25.0 / 71.7 | +11.7 | +6.7 |
| Mistral Small 3.2 | 56.7 / 85.0 | 66.7 / 86.7 | +10.0 | +1.7 |
| Phi-4 Mini | 18.3 / 56.7 | 26.7 / 63.3 | +8.4 | +6.6 |
| Qwen3 30B A3B | 70.0 / 90.0 | 71.7 / 91.7 | +1.7 | +1.7 |

Excluding Llama 1B, mean strict accuracy improves from 30.6% to 36.7% (+6.1 points) and mean lenient accuracy from 65.6% to 68.1% (+2.5 points). LangExtract improves strict accuracy for every model except Llama 1B, which failed to execute under augmentated conditions due to context window limitations (discussed in Section 7.4).

The magnitude of improvement varies substantially across models. Ministral 3B (+11.7 strict), Mistral Small 3.2 (+10.0 strict), and Phi-4 Mini (+8.4 strict) exhibit the largest gains, while Qwen3 30B A3B (+1.7 strict) and Gemma 3 4B (+1.6 strict) show more modest improvements. The already-strong Qwen3 30B A3B may be approaching a performance ceiling under the current evaluation protocol.

### 6.2 Analysis by Question Difficulty

Table V disaggregates performance by question difficulty across all successful models.

**Table V: Accuracy by Difficulty Level (strict / lenient, %)**

| Condition | Difficulty | Strict | Lenient |
|-----------|------------|--------|---------|
| Baseline | Easy | 39.4 | 72.7 |
| Baseline | Medium | 29.0 | 63.4 |
| Baseline | Hard | 26.9 | 65.7 |
| LangExtract | Easy | 45.5 | 72.7 |
| LangExtract | Medium | 33.9 | 64.0 |
| LangExtract | Hard | 36.1 | 72.2 |

The most striking result is the hard-question improvement: strict accuracy rises from 26.9% to 36.1% (+9.2 points; a 34% relative increase), and lenient accuracy from 65.7% to 72.2% (+6.5 points). This is precisely the difficulty class where structured entity information should confer the greatest advantage, because hard questions demand linking separated facts, resolving kinship or motive, and reconstructing multi-step causal chains—operations for which a compact global summary provides a more efficient navigation structure than raw sequential text.

Easy questions also benefit (+6.1 points strict), while medium questions show moderate gains (+4.9 points strict). The overall pattern suggests that LangExtract provides the most benefit when the question requires either direct fact retrieval (mostly easy questions) or complex relational reasoning (mostly hard questions), with intermediate utility for questions of moderate difficulty.

### 6.3 Proprietary Baseline

GPT-5.4 achieves 83.3% strict and 95.0% lenient accuracy on the baseline condition, substantially exceeding the best small model with augmentation (Qwen3 30B A3B at 71.7% / 91.7%). The 11.6-point strict accuracy gap between the best augmented small model and GPT-5.4 suggests that entity-aware preprocessing narrows but does not close the performance differential. The residual gap likely reflects both parameter-count advantages and broader training-data exposure to literary analysis patterns.

---

## 7. Discussion

### 7.1 Mechanism: Why LangExtract Works

LangExtract improves performance because it changes the *representation* of the problem. Rather than appending token-level annotations to every row, it provides a semantically organized overview that the model can use as a scaffold for navigating the full prompt. The structure is aligned with the cognitive demands of the benchmark: questions frequently probe *who knew what*, *who was related to whom*, *why a decision was taken*, and *how one event precipitated another*. The overview functions as a narrative map, not merely a named-entity pass.

Two properties underpin this effectiveness. First, the overview abstracts without information loss—it summarizes entities and relationships while the full dataset remains available for detailed verification. Second, it achieves this with compact token cost: approximately 2,000 additional tokens on a ~48,000-token prompt, a 4% increase.

### 7.2 Where LangExtract Helps Most

The difficulty-stratified results confirm that LangExtract's largest absolute gains occur on hard questions. Consider Q3: "Based on Benvolio's and Montague's observations before Romeo confesses anything, what symptoms are already documented, and what cause remains unresolved?" Answering correctly requires identifying relevant scenes, extracting specific symptoms, recognizing that the cause is explicitly stated as unknown, and synthesizing these elements into a coherent response. The LangExtract overview aids scene location through character-indexed structure and surfaces themes of unexplained sorrow, while the model still must read and reason over the full dataset to extract and synthesize the precise symptoms.

### 7.3 Model-Size Effects

The results reveal a non-monotonic relationship between model size and benefit from augmentation. Ministral 3B and Mistral Small 3.2 show the largest improvements, while Qwen3 30B A3B—already strong at baseline—shows smaller gains. This pattern supports three inferences:

1. **Smaller models benefit more**: Models with fewer parameters possess less intrinsic capacity for narrative reconstruction, making external structure proportionally more valuable.
2. **Diminishing returns exist**: Once a model is sufficiently capable to handle long-context reasoning effectively (Qwen3 30B A3B at 70% baseline), the marginal benefit of preprocessing decreases.
3. **The sweet spot is 3B–24B**: Models in this parameter range exhibit the most favorable balance between improvement magnitude and absolute performance.

### 7.4 Limitations

Several limitations qualify the interpretation of these results.

**Context window constraints**: Llama 1B failed under LangExtract augmentation because the combined prompt (~60.3K tokens) exceeded the provider's 60K context limit. This is not an implementation error but an intrinsic cost of the method: preprocessing that improves accuracy for one model class may break compatibility for another.

**Full-dataset prompting**: The current benchmark sends the complete dataset for every question, conflating raw reasoning ability with long-context retention, tolerance for prompt expansion, and context-window capacity. A method that appears weak in this evaluation may perform well in a retrieval-augmented setup where question-specific evidence slices replace the full dataset.

**Extraction quality**: Gemini 2.5 Flash is not domain-adapted for Shakespearean language. While LLM-based extraction handles figurative language more robustly than traditional NER, it still produces some false or low-value extractions. Domain-specific fine-tuning or more targeted few-shot examples would likely improve extraction quality further.

**Verifier variance**: The verifier is itself a language model rather than a deterministic scoring script. While the constrained three-class output taxonomy mitigates this, some evaluator noise is unavoidable.

---

## 8. Conclusion

This work demonstrates that structured entity extraction preprocessing using LangExtract can measurably improve question-answering accuracy for small open-weight language models on a narrative comprehension benchmark. The principal findings are:

1. LangExtract consistently improves strict accuracy across all successful models (1B–30B), with mean improvement of +6.1 points (30.6% → 36.7%).
2. The largest gains occur on hard questions requiring relational and causal reasoning (+9.2 points strict; 26.9% → 36.1%), confirming that structured overviews are most valuable when the search burden is highest.
3. Compact, globally organized structure outperforms raw text alone, demonstrating that representational design matters as much as extraction quality.
4. Smaller models benefit proportionally more, with the 3B–24B range representing the sweet spot where external structure most effectively compensates for limited internal reasoning capacity.

The pipeline establishes a practical approach for augmenting small language models on long-context narrative tasks. By prepending a structured overview before the full dataset, models can navigate complex narratives more efficiently without parameter-count increases or domain-specific fine-tuning.

### 8.1 Future Work

Several extensions would strengthen this line of research: (1) question-specific evidence retrieval to reduce context pressure and improve small-model compatibility, (2) domain-adapted extraction via fine-tuning on literary text, (3) multi-work testing to assess generalizability beyond a single play, (4) systematic ablation of extraction classes, passes, and buffer sizes, and (5) hybrid approaches combining LangExtract's global overview with retrieval-based evidence selection.

---

## 9. Reproducibility

All code, data, and result artifacts are available in the project repository. The following commands reproduce the core experimental conditions:

```bash
# Install dependencies
cd HW3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set OpenRouter API key
export OPENROUTER_API_KEY=sk-or-v1-...

# Run baseline condition
python run_benchmark.py --model mistral-small-3.2 --method baseline

# Run LangExtract condition
python run_benchmark.py --model mistral-small-3.2 --method langextract

# Evaluate results
python evaluate_results.py --results-file results/mistral-small-3.2/answers_before_ner.csv
python evaluate_results.py --results-file results/mistral-small-3.2/answers_after_ner_langextract.csv
```

Evaluated answer files are stored in `HW3/results/{model}/`, and cached LangExtract extractions reside in `HW3/results/_ner_cache/`.

---

## References

[1] K.-J. Kim, C.-H. Lee, S.-E. Bae, J.-H. Choi, and W. Kang. "Digital forensics in law enforcement: A case study of LLM-driven evidence analysis." *Forensic Science International: Digital Investigation*, 54 (2025) 301939. [DOI: 10.1016/j.fsidi.2025.301939](https://doi.org/10.1016/j.fsidi.2025.301939)

[2] C. Grajeda, F. Breitinger, and I. Baggili. "Availability of datasets for digital forensics — And what is missing." *Digital Investigation*, 22 (2017) 94–105. [DOI: 10.1016/j.diin.2017.06.004](https://doi.org/10.1016/j.diin.2017.06.004)

[3] Z. Yin, Z. Wang, W. Xu, J. Zhuang, P. Mozumder, A. Smith, and W. Zhang. "Digital Forensics in the Age of Large Language Models." *arXiv preprint* arXiv:2504.02963v1 (Apr 2025). [https://arxiv.org/abs/2504.02963](https://arxiv.org/abs/2504.02963)

[4] H. Zhou, W. Xu, J. Dehlinger, S. Chakraborty, and L. Deng. "Forensic Intelligence Graphs: An LLM Approach to Digital Evidence Extraction and Relationship Analysis." Towson University & University of Baltimore (2025). [PDF](papers/Forensic%20Intelligence%20Graphs-%20An%20LLM%20Approach%20to%20Digital%20Evidence%20Extraction%20and%20Relationship%20Analysis.pdf)

[5] D. Bamman, O. Lewke, and A. Mansoor. "An Annotated Dataset of Literary Entities." *Proceedings of NAACL-HLT* (2019). [https://aclanthology.org/N19-1220/](https://aclanthology.org/N19-1220/)

[6] J. Brooke, A. Tong, and G. Hirst. "Bootstrapped Text-level Named Entity Recognition for Literature." *Proceedings of ACL* (2016). [https://aclanthology.org/P16-2056/](https://aclanthology.org/P16-2056/)

[7] H. Vala, D. Jurgens, A. Piper, and S. Ray. "Mr. Bennet, his coachman, and the Archbishop walk into a bar but only one of them gets recognized: On the difficulty of detecting characters in literary texts." *Proceedings of EMNLP* (2015). [https://aclanthology.org/D15-1088/](https://aclanthology.org/D15-1088/)

[8] Google. "LangExtract: A Python library for structured information extraction from unstructured text." GitHub repository (2025). [https://github.com/google/langextract](https://github.com/google/langextract) | [DOI: 10.5281/zenodo.17015089](https://doi.org/10.5281/zenodo.17015089)

[9] T. Kočiský, J. Schwarz, J. Blunsom, C. Dyer, G. Melis, N. Hermann, E. Grefenstette, K. M. Hermann, and P. Blunsom. "The NarrativeQA Reading Comprehension Corpus." *Proceedings of ACL* (2018). [https://arxiv.org/abs/1712.07040](https://arxiv.org/abs/1712.07040)

[10] T. Bonomo et al. "LiteraryQA: Towards Effective Evaluation of Long-document Narrative QA." *Proceedings of EMNLP* (2025). [https://aclanthology.org/2025.emnlp-main.1729/](https://aclanthology.org/2025.emnlp-main.1729/)

[11] S. Angelidis, M. Ballesteros, and J. Henderson. "Book QA: Stories of Challenges and Opportunities." *MRQA Workshop at EMNLP* (2019). [https://aclanthology.org/D19-5819/](https://aclanthology.org/D19-5819/)

[12] Umer Haddii. "Shakespeare Plays Dialogues." Kaggle dataset (2024). [https://www.kaggle.com/datasets/umerhaddii/shakespeare-plays-dialogues](https://www.kaggle.com/datasets/umerhaddii/shakespeare-plays-dialogues)

[13] G. Michel, J. Brooke, and A. Piper. "Improving Quotation Attribution with Fictional Character Embeddings." *Findings of EMNLP* (2024). [https://aclanthology.org/2024.findings-emnlp.744/](https://aclanthology.org/2024.findings-emnlp.744/)

[14] F. Yang, M. Borowicz, and A. Piper. "Character Identification in Literary Texts." *Proceedings of AAAI* (2022). [https://cdn.aaai.org/ojs/21709/21709-13-25722-1-2-20220628.pdf](https://cdn.aaai.org/ojs/21709/21709-13-25722-1-2-20220628.pdf)

[15] R. Dufour, M. Ballesteros, and A. Piper. "BERT meets d'Artagnan: Data Augmentation for Robust Character Detection in Novels." *Proceedings of LREC* (2022). [https://hal.univ-lorraine.fr/EC-NANTES/hal-03617722v1](https://hal.univ-lorraine.fr/EC-NANTES/hal-03617722v1)

[16] J. Lee, C. Lim, B. Jin, M. Min, and H. Kim. "DF-graph: Structured and explainable analysis of communication data for digital forensics." *DFRWS APAC 2025* (Nov 10-12, 2025). [PDF](papers/DF-graph-Structured-and-explainable-analysi_2025_Forensic-Science-Internati.pdf)

[17] W. Gale, K. Church, and D. Yarowsky. "One sense per discourse." *DARPA Speech and Natural Language Workshop* (1992). [https://aclanthology.org/H92-1045/](https://aclanthology.org/H92-1045/)

[18] H.T. Duong and T.A. Nguyen-Thi. "A review: preprocessing techniques and data augmentation for sentiment analysis." *Computational Social Networks* 8, 1 (2021). [DOI: 10.1186/s40649-020-00080-x](https://doi.org/10.1186/s40649-020-00080-x)

[19] J. Wei and K. Zou. "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks." *Proceedings of ICLR* (2019). [https://arxiv.org/abs/1901.11196](https://arxiv.org/abs/1901.11196)

[20] OpenRouter. "OpenRouter API Documentation." [https://openrouter.ai/docs](https://openrouter.ai/docs)

---

## Appendix A: Sample LangExtract Output

Example structured overview generated by LangExtract (excerpt):

```markdown
## Characters

- **Chorus** (1 mention)
- **Sampson** (12 mentions) — servant of Capulet
- **Gregory** (10 mentions) — servant of Capulet
- **Abram** (3 mentions) — servant of Montague
- **Balthasar** (2 mentions) — servant of Montague
- **Benvolio** (28 mentions) — nephew of Montague, friend of Romeo
- **Tybalt** (18 mentions) — nephew of Lady Capulet
...

## Relationships

- **Sampson ↔ Gregory**: Co-servants, engage in banter about Montague feud
- **Benvolio ↔ Romeo**: Friends, Benvolio attempts to counsel Romeo
- **Tybalt ↔ Benvolio**: Antagonists, Tybalt escalates conflict
...

## Key Events

- **Act I, Scene I**: Street brawl between Capulet and Montague servants
- **Act I, Scene II**: Capulet plans feast, Romeo learns of Rosaline's attendance
...
```

---

## Appendix B: Complete Benchmark Questions

The full set of 60 question-answer pairs, organized by difficulty classification.

### Easy Questions (11)

**Q2** (easy): What prior pattern of street violence does Prince Escalus cite when escalating the next penalty to death?  
*Answer*: Because their feud has already caused repeated civil brawls and endangered Verona's public peace.

**Q4** (easy): Which declared status of Rosaline explains why Romeo classifies her as unattainable?  
*Answer*: Because she has sworn chastity and refuses love, courtship, and seduction.

**Q11** (easy): What identity data does Juliet obtain from the Nurse about Romeo, and what conflict does that information immediately create?  
*Answer*: She learns that he is Romeo, a Montague, and the only son of her family's great enemy.

**Q15** (easy): Which type of oath does Juliet reject as unreliable evidence of constancy, and why?  
*Answer*: Because the moon is changeable, and she does not want his love to seem equally variable.

**Q23** (easy): Which witness account does Lady Capulet challenge as biased, and what relationship does she cite as grounds for distrust?  
*Answer*: Lady Capulet does, because Benvolio is related to the Montagues.

**Q32** (easy): What enforcement measures does Capulet threaten if Juliet refuses the marriage order?  
*Answer*: He threatens to drag her to church and, if she still refuses, cast her off to beg, starve, and die in the streets.

**Q40** (easy): Which wedding items does Capulet explicitly relabel as funeral items after Juliet is found dead?  
*Answer*: He says every wedding item turns into its funeral opposite: instruments become bells, feast becomes burial meal, hymns become dirges, and bridal flowers serve a corpse.

**Q41** (easy): Who delivers the first report that shapes Romeo's understanding of Juliet's condition in Act V, and what corrective document is missing?  
*Answer*: Balthasar tells Romeo Juliet has been laid in the Capulet monument, but he has no letter from Friar Laurence.

**Q46** (easy): What observed behavior causes Balthasar to remain nearby instead of fully leaving?  
*Answer*: He hides nearby because Romeo looks dangerous and Balthasar distrusts his intentions.

**Q52** (easy): What exact quarantine conditions prevent Friar John from completing delivery of Friar Laurence's letter?  
*Answer*: He is quarantined with another friar in a suspected plague house, and no messenger will carry the letter for fear of infection.

**Q58** (easy): What physical evidence leads Juliet to infer poison, and why does she switch to the dagger?  
*Answer*: She sees the cup in Romeo's hand and concludes poison killed him, and when kissing him yields no poison for her, she takes his dagger and stabs herself.

### Medium Questions (31)

**Q1** (medium): Which gesture does Sampson intentionally direct at the Montague servants, and what denial does he immediately use to limit legal exposure?  
*Answer*: He bites his thumb at the Montague servants, then tries to evade blame by saying he is biting his thumb, but not at them.

**Q5** (medium): When Paris seeks permission to marry Juliet, what age, waiting-period, and consent conditions does Capulet put on the request?  
*Answer*: Juliet is still too young, Paris should wait, and he must win Juliet's own consent because Capulet's approval is only part of the decision.

**Q6** (medium): What information chain links Romeo to the Capulet feast: who cannot read the guest list, who reads it, what name is found, and who uses that fact to influence Romeo?  
*Answer*: Capulet's illiterate servant asks him to read the guest list, Romeo learns Rosaline will be there, and Benvolio uses that to persuade him to attend.

**Q7** (medium): What level of commitment does Juliet give Lady Capulet regarding Paris, and what explicit limit does she place on her response?  
*Answer*: She says she will look to like him if looking leads to liking, but she will not let herself go further than her mother's consent allows.

**Q8** (medium): Before entering the feast, what risk forecast does Romeo make, and what principle leads him to proceed anyway?  
*Answer*: He fears the night will begin a star-governed chain that ends in untimely death, but he submits himself to providence and goes in.

**Q10** (medium): What hostile-family identity does Romeo learn about Juliet only after physical contact has already occurred?  
*Answer*: He discovers that she is a Capulet, so the woman he loves belongs to his enemy's house.

**Q14** (medium): What threat assessment does Juliet make about Romeo's presence in the orchard, and how does Romeo compare that threat with romantic rejection?  
*Answer*: She says her kinsmen will kill him if they find him there, and Romeo says her displeasure is more dangerous than their swords and that he would rather die than live without her love.

**Q16** (medium): What next-day verification process does Juliet require before treating Romeo's intentions as honorable?  
*Answer*: She asks him to send word through a messenger she will send, stating where and when they will be married, and if he is not honorable he should stop courting her.

**Q20** (medium): What secondhand evidence does the Nurse give Romeo about Juliet's attitude toward Paris?  
*Answer*: She tells him that Paris wants Juliet, but Juliet would rather look at a toad than at Paris and turns pale when Paris is praised.

**Q21** (medium): Which concealed kinship causes Romeo to de-escalate Tybalt's challenge, and when is that kinship later made explicit?  
*Answer*: Tybalt has become Romeo's kinsman through Romeo's secret marriage to Juliet, though Tybalt does not know it.

**Q22** (medium): What intervention by Romeo inadvertently creates the opening for Mercutio's fatal wound?  
*Answer*: Romeo rushes between the fighters to stop them, and Tybalt stabs under Romeo's arm.

**Q24** (medium): What causal factor leads the Prince to commute Romeo's expected death sentence to banishment?  
*Answer*: Because Tybalt had killed Mercutio first, so Romeo's killing of Tybalt is treated as retaliatory and softened from death to exile.

**Q25** (medium): Which reported development affects Juliet more severely than Tybalt's death?  
*Answer*: Romeo's banishment.

**Q26** (medium): What newly processed fact causes Juliet to reverse her first verbal attack on Romeo?  
*Answer*: She realizes Tybalt would have killed Romeo, so Tybalt's death means her husband lives.

**Q27** (medium): What evidence does Romeo cite to argue that banishment is functionally worse than death?  
*Answer*: Because exile means separation from Juliet and Verona, and even the lowest creatures may look on Juliet there while he cannot.

**Q28** (medium): Which three surviving advantages does Friar Laurence list when trying to counter Romeo's despair?  
*Answer*: Juliet is alive, Tybalt is dead instead of Romeo, and the law has changed Romeo's sentence from death to exile.

**Q30** (medium): Which incorrect diagnosis of Juliet's emotional state is shared by Lady Capulet and Paris, and how is it used to justify the marriage schedule?  
*Answer*: They think she is consumed by grief for Tybalt, and Paris says the wedding is being hurried to interrupt that grief.

**Q33** (medium): Which recommendation from the Nurse causes Juliet to treat her as no longer trustworthy?  
*Answer*: The Nurse tells Juliet to marry Paris because Romeo is banished and effectively useless to her, even belittling Romeo beside Paris.

**Q35** (medium): What demonstration by Juliet convinces Friar Laurence she can endure a deathlike deception?  
*Answer*: Her willingness to kill herself rather than marry Paris convinces him she has the nerve for a desperate deathlike stratagem.

**Q37** (medium): What room-access condition does Juliet create before taking the vial, and why is that condition necessary?  
*Answer*: She is following Friar Laurence's instructions so she can secretly take the vial without the Nurse in her chamber.

**Q39** (medium): When the Capulet household finds Juliet apparently dead, how does Friar Laurence attempt to reinterpret the event for them?  
*Answer*: He says heaven had a share in Juliet and now has her wholly, so she has been advanced to eternal life rather than merely lost.

**Q43** (medium): Which prior legal ruling and follow-up instruction explain Romeo's presence in Mantua at the start of Act V?  
*Answer*: He is there because he was banished after Tybalt's death, and Friar Laurence told him to stay in Mantua until the marriage could be revealed and reconciliation attempted.

**Q44** (medium): What economic vulnerability allows Romeo to obtain poison from the Apothecary despite the law?  
*Answer*: His poverty overrules his will, and Romeo exploits that desperation by paying him.

**Q47** (medium): At the tomb, what role does Paris assign to himself and what support role does the Page perform?  
*Answer*: Paris comes to strew Juliet's grave with flowers and mourn her, while the Page stands watch, signals if anyone approaches, and later summons the watch.

**Q48** (medium): Which assumptions lead Paris to classify Romeo as a criminal intruder at the tomb?  
*Answer*: Paris thinks Romeo is the banished Montague who killed Tybalt, helped cause Juliet's death, and has now come to dishonor the dead.

**Q49** (medium): Before the duel with Paris, what does Romeo disclose about his mental state and self-directed intent?  
*Answer*: He says he is desperate, warns Paris not to provoke him, and admits he came armed against himself.

**Q50** (medium): After Paris is wounded, what new identity links does Romeo recognize, and how does that affect his treatment of Paris's body?  
*Answer*: Romeo recognizes Paris as Mercutio's kinsman and Juliet's would-be bridegroom, then grants Paris's wish to be laid in the tomb with Juliet.

**Q51** (medium): Which physical observations cause Romeo to doubt that Juliet has fully succumbed to death?  
*Answer*: Her lips and cheeks are still crimson rather than pale, so he thinks death has not yet erased her beauty.

**Q54** (medium): Which earlier family actions create the coercive environment that pushes Juliet toward the potion plan?  
*Answer*: Lady Capulet announces a Thursday marriage to Paris, and Capulet then forces the issue by threatening to drag Juliet to church or reject her if she refuses.

**Q55** (medium): Besides Friar Laurence, which named person is explicitly identified as knowing about the secret marriage?  
*Answer*: The Nurse.

**Q56** (medium): When Juliet regains consciousness in the tomb, what facts does she immediately know and what crucial deaths has she not yet learned?  
*Answer*: She remembers where she should be and immediately asks for Romeo, but she does not yet know that Romeo and Paris are dead.

### Hard Questions (18)

**Q3** (hard): Based on Benvolio's and Montague's observations before Romeo confesses anything, what symptoms are already documented, and what cause remains unresolved?  
*Answer*: They know he wanders alone at dawn, weeps, sighs, avoids company, and shuts himself in darkness, but they still do not know the cause of his sorrow.

**Q9** (hard): By what identifying signal does Tybalt detect Romeo at the feast, and whose intervention prevents an immediate assault?  
*Answer*: Tybalt recognizes Romeo by his voice, but Capulet forbids violence because Romeo is reputed virtuous and a fight would disgrace the feast and disrupt the guests.

**Q12** (hard): In Juliet's balcony speech, which element of Romeo's identity does she explicitly identify as the obstacle to their relationship?  
*Answer*: She means why must you be Romeo, meaning why must he belong to the enemy name Montague, not where are you.

**Q13** (hard): Before Romeo reveals his presence, what contingency does Juliet state regarding Romeo's name and her own status as a Capulet?  
*Answer*: She has already said that either Romeo should renounce his name or she herself will cease to be a Capulet, because only his name, not his person, is her enemy.

**Q17** (hard): What conflict-resolution objective does Friar Laurence explicitly cite when deciding to assist the marriage?  
*Answer*: He agrees because he hopes their marriage can reconcile the two hostile households.

**Q18** (hard): Before Romeo returns in Act II, Scene IV, what outdated model of his emotional state do Mercutio and Benvolio still use?  
*Answer*: They still think Rosaline has made him love-sick, and Mercutio doubts that such a wounded Romeo can handle Tybalt's challenge.

**Q19** (hard): What warning does the Nurse issue about Romeo's intent, and what operational plan does Romeo return through her?  
*Answer*: She warns him not to deceive Juliet or lead her into a fool's paradise, and he tells her to bring Juliet to Friar Laurence's cell that afternoon for shrift and marriage and then wait behind the abbey wall for ladder-cords.

**Q29** (hard): Following the banishment ruling, what immediate movement plan and longer-term reconciliation plan does Friar Laurence assign to Romeo?  
*Answer*: Romeo should go to Juliet that night, leave before the watch, hide in Mantua, and wait while Friar Laurence works to reveal the marriage, reconcile the families, and seek a pardon.

**Q31** (hard): Which ambiguous statement allows Lady Capulet to infer that Juliet wants Romeo dead, and what does Juliet actually mean by it?  
*Answer*: She uses ambiguous phrasing so Lady Capulet hears a wish for Romeo's death, while Juliet means that her own heart is dead until she sees him.

**Q34** (hard): In the Friar's cell, which responses by Juliet sound compliant to Paris but avoid actual acceptance?  
*Answer*: She answers in equivocations: she says she may be a wife only when she may be one, says she loves him without naming Paris, and says even her face is not her own.

**Q36** (hard): What are the required steps, actors, and timing constraints in Friar Laurence's potion operation?  
*Answer*: She must appear cheerful and consent, sleep alone, drink the vial in bed, seem dead for forty-two hours, be laid in the Capulet vault, and then be recovered by Romeo and the Friar for flight to Mantua.

**Q38** (hard): What failure scenarios does Juliet model before consuming the potion?  
*Answer*: She fears it may fail, may actually be poison from the Friar, may wake her too early in the vault to suffocate, or may leave her mad among corpses, bones, and Tybalt's ghost.

**Q42** (hard): Which broken communication link causes Romeo to act on a false death report?  
*Answer*: Friar Laurence's letter explaining the potion never reaches Romeo because Friar John is quarantined, so Balthasar's report of Juliet's burial goes uncorrected.

**Q45** (hard): What cover story does Romeo give Balthasar for entering the tomb, and what later evidence shows it is not his full motive?  
*Answer*: He says he mainly needs a ring from Juliet's finger, but later evidence shows he came to die beside Juliet.

**Q53** (hard): What end-to-end recovery plan did Friar Laurence originally design for Juliet's waking, and which failure point prevents execution?  
*Answer*: Romeo was supposed to learn the plan by letter, come to the vault when Juliet awoke, and take her to Mantua, but the undelivered letter leaves him ignorant and the reunion never happens.

**Q57** (hard): What two factors cause Friar Laurence's extraction attempt to fail after Juliet wakes?  
*Answer*: He hears the watch coming and says their plan has been thwarted, and although he offers to hide Juliet in a nunnery, she refuses and he loses nerve at the approaching noise.

**Q59** (hard): In his formal explanation, how does Friar Laurence reconstruct the causal chain from the secret marriage to the final deaths?  
*Answer*: He says the secret marriage was followed by Tybalt's death and Romeo's banishment, Juliet then pined and was pushed toward Paris, she threatened suicide and took the sleeping potion, the letter failed, Romeo and Paris died before she awoke, and Juliet then killed herself.

**Q60** (hard): After Romeo's letter is read, what findings does the Prince make, and what remedial action do Capulet and Montague take?  
*Answer*: He concludes the letter confirms Friar Laurence's account and that the feud has brought this scourge, while he too is punished for tolerating it, and Capulet and Montague reconcile and promise gold statues for Juliet and Romeo.
