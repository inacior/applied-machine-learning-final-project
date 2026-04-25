# A hybrid NER–SLM pipeline for entity-aware narrative question answering

## Context

Structured entity extraction is a critical capability in both digital forensics and narrative comprehension, where identifying who did what, to whom, and under which relationships can determine whether an analysis is actionable. In digital forensics, large language models (LLMs) have been applied to messenger data for evidence triage and entity extraction, demonstrating improved recall over keyword-only approaches [1], but reproducibility and dataset availability remain persistent obstacles [2]. Recent surveys further caution that LLM outputs can be difficult to validate without grounding in source evidence, especially for small, open-weight models [3]. Meanwhile, prior work on forensic intelligence graphs has shown that accumulating structured entity extractions into a navigable graph supports downstream analysis and review [4].

This project investigates whether lightweight, deterministic entity extraction preprocessing can improve the question-answering accuracy of small open-weight language models (2B–7B parameters) on a forensics-style narrative comprehension benchmark.

## Objectives

1. Develop a reproducible preprocessing pipeline that applies named entity recognition to the complete text of a literary work, producing structured entity annotations and a relationship graph grounded in the source text.
2. Construct a benchmark of 60 question-answer pairs spanning factoid, relational, and causal questions, with per-question difficulty labels and answer validation references.
3. Evaluate multiple open-weight small language models (2B, 7B) via OpenRouter on this benchmark, comparing accuracy with and without the structured entity context.
4. Quantify the performance delta attributable to entity-aware preprocessing and identify which question types benefit most from structured entity information.
5. Benchmark the hybrid pipeline against a stronger proprietary LLM to estimate the remaining gap.

## Methodology

**Data and unit of analysis.** The primary dataset is the complete text of a dramatic work containing approximately 25,000 words across multiple acts and scenes, with speaker-attributed dialogue, stage directions, and scene metadata. The evaluation benchmark consists of 60 question-answer pairs organized as a forensics-style evidence questionnaire, classified by difficulty (easy, medium, hard) and linked to specific line ranges for answer validation.

**Pipeline.** The workflow has three stages. First, a named entity recognition stage processes the raw text to extract entities — characters, their relationships, and contextual metadata — producing a structured entity graph and per-entity profiles. Second, for each question in the benchmark, small language models are queried via OpenRouter in two configurations: (a) raw text only (the relevant portion of the source text) and (b) entity-augmented context (structured entity annotations prepended to the raw text). Open-weight models in the 2B and 7B parameter range are selected so results remain reproducible and self-hostable. Each query follows a standardised prompt template requesting concise, evidence-grounded answers.

**Evaluation.** Accuracy is assessed through both exact and semantic match against ground-truth answers, with results stratified by question difficulty and question type. We report per-model accuracy, the net improvement from entity augmentation, and token-cost comparisons between raw and augmented contexts. An ablation teases apart the contribution of different entity information types. A proprietary frontier LLM is evaluated identically to establish a performance ceiling.

## Expected results

Expected deliverables are: (i) a reusable entity extraction and graph construction pipeline for narrative texts; (ii) a structured entity graph for the source material; and (iii) a systematic accuracy comparison across model scales with and without entity augmentation.

We hypothesize that structured entity context will yield the largest gains on identity and relationship questions while offering marginal benefit on purely thematic or interpretive items. We further expect that 7B-class models with entity augmentation may narrow the gap to the proprietary baseline on factoid questions, demonstrating a cost-effective and reproducible alternative for entity-heavy question answering.

## References

[1] K.-J. Kim, C.-H. Lee, S.-E. Bae, J.-H. Choi, and W. Kang. *Digital forensics in law enforcement: A case study of LLM-driven evidence analysis.* Forensic Science International: Digital Investigation, 54 (2025) 301939. DOI: 10.1016/j.fsidi.2025.301939.

[2] C. Grajeda, F. Breitinger, and I. Baggili. *Availability of datasets for digital forensics — And what is missing.* Digital Investigation, 22 (2017) 94–105. DOI: 10.1016/j.diin.2017.06.004.

[3] Z. Yin, Z. Wang, W. Xu, J. Zhuang, P. Mozumder, A. Smith, and W. Zhang. *Digital Forensics in the Age of Large Language Models.* arXiv:2504.02963v1 (Apr 2025).

[4] H. Zhou, W. Xu, J. Dehlinger, S. Chakraborty, and L. Deng. *Forensic Intelligence Graphs: An LLM Approach to Digital Evidence Extraction and Relationship Analysis.* 2025.
