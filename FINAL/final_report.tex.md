\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{url}
\usepackage{hyperref}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, calc}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{Entity-Aware Narrative Question Answering:\\
A LangExtract-Augmented Pipeline for Small Language Models}

\author{\IEEEauthorblockN{Renan In\'{a}cio}
\IEEEauthorblockA{\textit{Federal University of Santa Catarina}\\
Florian\'{o}polis, Brazil \\
inacior.dev@gmail.com}
\and
\IEEEauthorblockN{Th\'{e}a Louise Sequeira Pessoa}
\IEEEauthorblockA{\textit{Federal University of Santa Catarina}\\
Florian\'{o}polis, Brazil \\
theapessoa@gmail.com}
\and
\IEEEauthorblockN{Pedro Henrique Cavalcante S\'{a}}
\IEEEauthorblockA{\textit{Federal University of Santa Catarina}\\
Florian\'{o}polis, Brazil \\
pedroccavalcante@gmail.com}
}

\maketitle

\begin{abstract}
Structured entity extraction is a cornerstone capability in digital forensics and narrative comprehension, where identifying actors, relationships, and causal chains determines whether an analysis is actionable. This work investigates whether compact, globally organized entity summaries produced by Google's LangExtract library can improve question-answering accuracy for small open-weight language models (1B--30B parameters) on a narrative comprehension benchmark. Using Shakespeare's \textit{Romeo and Juliet} as a test corpus---a work whose dense dialogue, archaic diction, and intricate character relationships make it a demanding proxy for forensic narrative analysis---we construct a 1,068-row preprocessed dataset and a 60-question evaluation set spanning three difficulty levels. LangExtract, with Gemini 2.5 Flash as the extraction backend, produces a semantic overview of characters, relationships, events, locations, and themes, which is prepended to the full dataset before querying seven small language models via OpenRouter. Excluding the context-limited Llama~1B, LangExtract improves mean strict accuracy from 30.6\% to 36.7\% (+6.1 points) and lenient accuracy from 65.6\% to 68.1\% (+2.5 points). The largest gains materialize on hard questions requiring relational and causal reasoning (26.9\% $\rightarrow$ 36.1\% strict; +9.2 points). These results demonstrate that compact, globally organized entity structure reduces narrative search burden more effectively than raw text alone, particularly for models in the 3B--24B parameter range where internal reasoning capacity remains constrained.
\end{abstract}

\begin{IEEEkeywords}
named entity recognition, small language models, structured extraction, LangExtract, narrative question answering, long-context reasoning, LLM-augmented preprocessing
\end{IEEEkeywords}

\section{Introduction}

\subsection{Context and Motivation}

The extraction of structured entities from unstructured text is increasingly central to both digital forensics and computational literary analysis. In forensic investigation, large language models have been applied to messenger data for evidence triage and entity identification, demonstrating improved recall over keyword-only approaches~\cite{b1}. However, reproducibility remains fragile---dataset availability is a persistent bottleneck~\cite{b2}---and recent surveys caution that LLM outputs are difficult to validate without grounding in source evidence, particularly for small, open-weight models~\cite{b3}. Forensic intelligence graphs address some of these concerns by accumulating structured entity extractions into navigable representations that support downstream analysis and review~\cite{b4}.

These findings suggest a broader principle with significance beyond forensics: if lightweight, deterministic entity extraction can produce a structured overview of a narrative, it may improve the question-answering accuracy of small language models that lack the parameter budget for internal reconstruction of complex relational structures from raw text alone.

\subsection{Problem Definition}

Long-context question answering over literary corpora is challenging for small and mid-sized language models because the task combines multiple cognitive burdens simultaneously: retaining extensive textual context, identifying which passages bear on a given question, linking temporally and spatially separated events, and reconstructing character relationships without drifting into unsupported inference. A Shakespeare play constitutes an especially rigorous test bed. Dialogue is dense and stylized, character names appear in multiple morphological forms, stage directions encode important state transitions, and the Elizabethan lexicon can mislead general-purpose named entity systems.

The central empirical question is whether explicit entity-aware preprocessing can mitigate these burdens. If a model first receives a structured overview of characters, events, and relationships before processing the full dataset, it may navigate the long prompt more efficiently and generate more accurate answers. This hypothesis is most consequential for questions about identity, kinship, scene participation, and causal chains---precisely the categories where small models most frequently fail.

\subsection{Objectives}

The specific objectives of this work are:

\begin{enumerate}
\item To develop a reproducible preprocessing pipeline that applies structured entity extraction to the complete text of a literary work, producing a grounded entity overview.
\item To construct a benchmark of 60 question-answer pairs spanning factoid, relational, and causal types, with per-question difficulty labels and evidence-level validation.
\item To evaluate multiple open-weight small language models (1B--30B parameters) via OpenRouter, comparing accuracy with and without structured entity context.
\item To quantify the performance delta attributable to entity-aware preprocessing and identify which question types benefit most.
\item To benchmark the hybrid pipeline against a stronger proprietary LLM to estimate the residual performance gap.
\end{enumerate}

\section{Related Work}

\subsection{Named Entity Recognition for Literary Texts}

The domain-shift problem in NER is well documented. Standard models trained on news corpora underperform sharply on literary text. Bamman et al.~\cite{b5} reported that ACE 2005-trained models drop from 68.8 F1 on news to 45.7 F1 on literary text---a 23-point decline. Literature exhibits proportionally more person and facility entities, fewer organization and geopolitical entities, and 13.8\% of literary entities contain nested structures that flat sequence-labeling NER cannot capture. Brooke et al.~\cite{b6} found that Stanford CoreNLP achieved only 0.751 F-measure on fiction, versus 0.792 for a bootstrapped domain-specific approach. Vala et al.~\cite{b7} further showed that off-the-shelf NER omits titles from names and fails to recognize characters referenced by common nouns, such as ``the Nurse'' or ``the Friar.''

These domain-shift findings motivated our choice of an LLM-based extraction approach (LangExtract) over traditional sequence-labeling NER. Unlike fixed-label classifiers, LLMs can exploit context to handle the figurative language, metonymy, and honorific patterns characteristic of literary text.

\subsection{LLM-Based Structured Extraction}

Google's LangExtract~\cite{b8} is a Python library for structured information extraction from unstructured text using LLMs as backend extractors. It differs from traditional NER in several respects: extraction classes are defined through natural language prompts and few-shot examples rather than a fixed tag schema; each extracted span is grounded to precise character offsets in the source, ensuring traceability; and long documents are divided into overlapping buffers with configurable chunking and parallel processing parameters.

Key architectural features include: \textbf{precise source grounding} (exact character offsets), a \textbf{chunking strategy} for long documents (\textit{max\_char\_buffer} controls chunk size), \textbf{multiple extraction passes} for iterative refinement, \textbf{parallel processing} with configurable worker counts, and \textbf{schema enforcement} through few-shot examples. LangExtract is compatible with multiple LLM backends, including Gemini, OpenAI, and custom providers via OpenRouter, and has been applied in medical information extraction tasks~\cite{b8}.

\subsection{Narrative Question Answering}

Recent years have seen the development of several narrative QA benchmarks. NarrativeQA~\cite{b9} provides approximately 46,000 question-answer pairs across 1,567 stories. LiteraryQA~\cite{b10} refines this approach with cleaned, validated questions specifically targeting literary works. BookQA~\cite{b11} frames character identification as a classification task. A consistent finding across these benchmarks is that long-context reasoning remains extremely difficult for models below 10B parameters, particularly for questions requiring linkage of information from distant parts of a narrative.

Our work differs in two respects: we focus on a single work with broad question coverage (60 questions spanning the entire play), and we explicitly test whether preprocessing with structured entity extraction improves performance, rather than measuring raw model capability in isolation.

\section{Dataset}

\subsection{Source Material}

The primary corpus is the complete text of Shakespeare's \textit{Romeo and Juliet}, sourced from the Kaggle dataset \textit{umerhaddii/shakespeare-plays-dialogues}~\cite{b12}. The original CSV provides dialogue lines with speaker attribution, act and scene metadata, and sequential line numbers.

\subsection{Preprocessing Pipeline}

The raw dataset was transformed through three principal preprocessing stages:

\textbf{Row consolidation} merged consecutive lines by the same character within a scene into single dialogue blocks, reducing the dataset from approximately 1,500 speech segments to 1,068 rows while preserving narrative continuity.

\textbf{Participant tracking} added a \textit{participants} column that enumerates characters physically present on stage at each moment, inferred from Enter, Exit, and Exeunt stage directions. This column enables interaction analysis and serves as a validation resource for ground-truth answers.

\textbf{Name normalization} decomposed character names into \textit{title}, \textit{first\_name}, \textit{last\_name}, and \textit{normalized\_name} columns. This handled honorifics (e.g., ``Lady Capulet,'' ``Friar Laurence'') and aliases, producing a canonical name that could be used consistently across the dataset.

The final dataset comprises 10 columns, detailed in Table~\ref{tab:schema}.

\begin{table}[htbp]
\caption{Dataset Schema}
\label{tab:schema}
\begin{center}
\begin{tabular}{|l|p{5.5cm}|}
\hline
\textbf{Column} & \textbf{Description} \\
\hline
\textit{act} & Act I through Act V \\
\hline
\textit{scene} & Prologue, Scene I--V \\
\hline
\textit{character} & Speaker name or \textit{[stage direction]} \\
\hline
\textit{dialogue} & Merged dialogue text \\
\hline
\textit{line\_number} & Last sequential line number in merged block \\
\hline
\textit{participants} & Characters physically on stage \\
\hline
\textit{title} & Extracted honorific (LADY, PRINCE, NURSE) \\
\hline
\textit{first\_name} & First name component \\
\hline
\textit{last\_name} & Last name component (CAPULET, MONTAGUE) \\
\hline
\textit{normalized\_name} & Canonical name string \\
\hline
\end{tabular}
\end{center}
\end{table}

\subsection{Benchmark Questions}

A set of 60 question-answer pairs was constructed as a forensics-style evidence questionnaire, classified by difficulty as shown in Table~\ref{tab:diff}.

\begin{table}[htbp]
\caption{Question Difficulty Distribution}
\label{tab:diff}
\begin{center}
\begin{tabular}{|l|c|p{4.5cm}|}
\hline
\textbf{Difficulty} & \textbf{Count} & \textbf{Description} \\
\hline
Easy & 11 & Direct factoid questions requiring single-fact recall \\
\hline
Medium & 31 & Relational and interpretive questions spanning multiple lines \\
\hline
Hard & 18 & Causal and multi-step reasoning questions requiring cross-scene linkage \\
\hline
\end{tabular}
\end{center}
\end{table}

Each question includes a ground-truth answer, analyst observations with line-range validation references, and CSV row indices for evidence reconstruction. The full question set is provided in Appendix~B.

\subsection{Data Quality Challenges}

Several domain-specific phenomena complicated both preprocessing and entity extraction:

\textbf{Markup artifacts}: The raw text contained line-number anchors (\textit{\{\#1.1.1\}}), speaker metadata tags (\textit{[**SAMPSON**]\{\#speech1\}}), and escape sequences requiring cleanup prior to NLP processing.

\textbf{Character aliases}: Romeo appears as ``son of Montague,'' ``my cousin,'' and ``villain Romeo'' at different points. Family affiliations (Capulet versus Montague) are not explicitly tagged in the source.

\textbf{Figurative language}: Metaphor (``It is the east, and Juliet is the sun''), personification (``Death is my son-in-law''), and metonymy (``My house and welcome on their pleasure stay,'' where ``house'' denotes household/family) all complicate entity extraction.

\textbf{Implicit references}: Pronouns and anaphoric references require coreference resolution to map to the correct character, a problem compounded by the absence of explicit pronoun-antecedent links in the raw text.

These challenges are precisely the type that LLM-based extraction (LangExtract) is better equipped to handle than traditional sequence-labeling NER, as discussed in Section~\ref{sec:method}.

\section{Methodology}\label{sec:method}

\subsection{Pipeline Overview}

The workflow proceeds in three stages:

\begin{enumerate}
\item \textbf{Entity extraction}: LangExtract processes the formatted dataset text to extract structured entities across seven classes.
\item \textbf{Context augmentation}: The extracted entities are organized into a compact semantic overview and prepended to the full dataset.
\item \textbf{Question answering}: Small language models receive the augmented context and answer each benchmark question. A verifier LLM evaluates answers against ground truth.
\end{enumerate}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=0.45cm and 0.35cm,
    box/.style={draw, rounded corners=1.5pt, fill=blue!5, text width=1.25cm,
                align=center, font=\scriptsize, minimum height=0.5cm, inner sep=2pt},
    arrow/.style={-{Stealth[scale=0.55]}, semithick},
]

% ── Row 1: Dataset preparation ──
\node[box, text width=1.05cm] (csv)  {Dataset\\CSV};
\node[box, right=of csv]      (fmt)  {Format as\\text};
\node[box, right=of fmt]      (le)   {LangExtract};
\node[box, right=of le]       (ov)   {Augmented Context};

% ── Row 2: Question answering ──
\node[box, text width=1.05cm,
      below=0.75cm of csv]    (qcsv) {Questions\\CSV};
\node[box, right=of qcsv]     (qfmt) {Format\\prompt};
\node[box, right=of qfmt]     (slm)  {SLM};
\node[box, right=of slm]      (ans)  {Answers};
\node[box, right=of ans]      (ver)  {Verifier};

% ── Horizontal arrows ──
\foreach \a/\b in {csv/fmt, fmt/le, le/ov, qcsv/qfmt, qfmt/slm, slm/ans, ans/ver}
  \draw[arrow] (\a) -- (\b);

% ── Vertical: overview feeds into SLM ──
\draw[arrow] (ov.south) -- ++(0,-0.2) -| (slm);

\end{tikzpicture}
\caption{Benchmark pipeline architecture. The LangExtract entity overview is
prepended to the prompt before querying the SLM.}
\label{fig:pipeline}
\end{figure}

\subsection{LangExtract Configuration}

The extraction pipeline uses \textit{google/gemini-2.5-flash} via OpenRouter as the extraction backend with seven entity classes: \textit{character}, \textit{relationship}, \textit{emotion}, \textit{event}, \textit{location}, \textit{object}, and \textit{theme}. The extraction prompt instructs the model to use exact source text for extraction spans, extract in order of appearance, and provide meaningful attributes for context. A single few-shot example demonstrates character and theme extraction from the Prologue.

Processing parameters are: \textit{extraction\_passes}=1, \textit{max\_workers}=5, and \textit{max\_char\_buffer}=50,000.

\subsection{Structured Overview Generation}

Raw LangExtract output is filtered to retain only grounded spans---extractions whose text appears verbatim in the source---and reorganized into a markdown-formatted overview comprising seven sections: Characters (with mention counts), Relationships (interactions and family affiliations), Key Events (organized by act and scene), Emotional Themes, Locations, Significant Objects, and Themes.

The resulting prompt structure appends this overview ($\sim$2,000 tokens) before the full dataset ($\sim$48,000 tokens) with instructions to use the overview as a navigation guide. A sample overview excerpt is provided in Appendix~A.

\subsection{Question Answering Protocol}

Each model receives the augmented context and answers all 60 questions under a system prompt requiring answers drawn solely from the provided context, paraphrased rather than quoted, with honest admission of insufficient information where applicable. Generation uses \textit{temperature}=0.0 for deterministic reproducibility, \textit{max\_tokens}=2,048, and rate-limited API calls (1.0-second delay, 3 retries with 1.0-second backoff).

\subsection{Evaluation Framework}

Answers are evaluated by a verifier LLM (\textit{openai/gpt-5.4-mini}) using a three-class taxonomy:

\begin{itemize}
\item \textbf{CORRECT}: Captures all essential facts from ground truth, no factual errors.
\item \textbf{PARTIALLY\_CORRECT}: Captures some key facts but misses important elements, or contains minor factual errors.
\item \textbf{INCORRECT}: Substantially wrong, contradicts ground truth, fabricates information, or fails to answer.
\end{itemize}

The verifier receives the question, ground-truth answer, analyst observations, reconstructed evidence rows, and the model's answer. Two composite metrics are reported: \textbf{strict accuracy} (percentage CORRECT) and \textbf{lenient accuracy} (percentage CORRECT or PARTIALLY\_CORRECT).

\section{Experimental Setup}

\subsection{Models Evaluated}

Seven open-weight small language models spanning 1B to 30B parameters were evaluated via OpenRouter, as detailed in Table~\ref{tab:models}. A proprietary baseline (GPT-5.4) was also run to establish the performance ceiling.

\begin{table*}[htbp]
\caption{Models Evaluated}
\label{tab:models}
\begin{center}
\begin{tabular}{|l|c|c|l|}
\hline
\textbf{Model} & \textbf{Parameters} & \textbf{Context} & \textbf{OpenRouter ID} \\
\hline
Gemma 3 4B & 4B & 32K & \textit{google/gemma-3-4b-it} \\
\hline
Llama 1B & 1B & 60K & \textit{meta-llama/llama-3.2-1b-instruct} \\
\hline
Llama 3B & 3B & 128K & \textit{meta-llama/llama-3.2-3b-instruct} \\
\hline
Ministral 3B & 3B & 128K & \textit{mistralai/ministral-3b-2512} \\
\hline
Mistral Small 3.2 & 24B & 128K & \textit{mistralai/mistral-small-3.2-24b-instruct-2506} \\
\hline
Phi-4 Mini & 3.8B & 128K & \textit{microsoft/phi-4-mini-instruct} \\
\hline
Qwen3 30B A3B & 30B${}^*$ & 128K & \textit{qwen/qwen3-30b-a3b-instruct-2507} \\
\hline
\multicolumn{4}{l}{${}^*$3B active parameters (mixture-of-experts architecture).} \\
\end{tabular}
\end{center}
\end{table*}

\subsection{Conditions Compared}

Two conditions were compared: a \textbf{baseline} consisting of raw formatted CSV rows with no entity extraction, and a \textbf{LangExtract} condition in which the structured semantic overview is prepended to the full dataset. This minimal-comparison design isolates the effect of entity-aware preprocessing.

\subsection{Implementation}

The benchmark system is implemented in Python. Key components include \textit{run\_benchmark.py} (orchestration), \textit{bench/ner\_pipeline.py} (LangExtract extraction and overview generation), \textit{bench/data\_loader.py} (dataset and question loading), \textit{bench/openrouter.py} (API client with rate limiting and retry logic), \textit{bench/evaluator.py} (LLM-based answer evaluation), and \textit{evaluate\_results.py} (standalone evaluation). All code, data, and result artifacts are available in the project repository.\footnote{\href{https://github.com/inacior/Entity-Aware-Narrative-Question-Answering-A-LangExtract-Augmented-Pipeline-for-Small-Language-Models}{GitHub repository}}

\section{Results}

\subsection{Per-Model Comparison}

Table~\ref{tab:permodel} reports strict and lenient accuracy for each model under both conditions.

\begin{table*}[htbp]
\caption{Per-Model Accuracy (strict / lenient, \%)}
\label{tab:permodel}
\begin{center}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Baseline} & \textbf{LangExtract} & $\Delta$ \textbf{Strict} & $\Delta$ \textbf{Lenient} \\
\hline
Gemma 3 4B & 11.7 / 43.3 & 13.3 / 45.0 & +1.6 & +1.7 \\
\hline
Llama 1B & 0.0 / 3.3 & 0.0 / 0.0 & 0.0 & $-$3.3 \\
\hline
Llama 3B & 13.3 / 53.3 & 16.7 / 50.0 & +3.4 & $-$3.3 \\
\hline
Ministral 3B & 13.3 / 65.0 & 25.0 / 71.7 & +11.7 & +6.7 \\
\hline
Mistral Small 3.2 & 56.7 / 85.0 & 66.7 / 86.7 & +10.0 & +1.7 \\
\hline
Phi-4 Mini & 18.3 / 56.7 & 26.7 / 63.3 & +8.4 & +6.6 \\
\hline
Qwen3 30B A3B & 70.0 / 90.0 & 71.7 / 91.7 & +1.7 & +1.7 \\
\hline
\end{tabular}
\end{center}
\end{table*}

Excluding Llama~1B, mean strict accuracy improves from 30.6\% to 36.7\% (+6.1 points) and mean lenient accuracy from 65.6\% to 68.1\% (+2.5 points). LangExtract improves strict accuracy for every model except Llama~1B, which failed to execute under augmentated conditions due to context window limitations (discussed in Section~\ref{sec:discuss}).

The magnitude of improvement varies substantially across models. Ministral~3B (+11.7 strict), Mistral Small~3.2 (+10.0 strict), and Phi-4 Mini (+8.4 strict) exhibit the largest gains, while Qwen3 30B A3B (+1.7 strict) and Gemma 3 4B (+1.6 strict) show more modest improvements. The already-strong Qwen3 30B A3B may be approaching a performance ceiling under the current evaluation protocol.

\subsection{Analysis by Question Difficulty}

Table~\ref{tab:difficulty} disaggregates performance by question difficulty across all successful models.

\begin{table}[htbp]
\caption{Accuracy by Difficulty Level (strict / lenient, \%)}
\label{tab:difficulty}
\begin{center}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Condition} & \textbf{Difficulty} & \textbf{Strict} & \textbf{Lenient} \\
\hline
Baseline & Easy & 39.4 & 72.7 \\
Baseline & Medium & 29.0 & 63.4 \\
Baseline & Hard & 26.9 & 65.7 \\
\hline
LangExtract & Easy & 45.5 & 72.7 \\
LangExtract & Medium & 33.9 & 64.0 \\
LangExtract & Hard & 36.1 & 72.2 \\
\hline
\end{tabular}
\end{center}
\end{table}

The most striking result is the hard-question improvement: strict accuracy rises from 26.9\% to 36.1\% (+9.2 points; a 34\% relative increase), and lenient accuracy from 65.7\% to 72.2\% (+6.5 points). This is precisely the difficulty class where structured entity information should confer the greatest advantage, because hard questions demand linking separated facts, resolving kinship or motive, and reconstructing multi-step causal chains---operations for which a compact global summary provides a more efficient navigation structure than raw sequential text.

Easy questions also benefit (+6.1 points strict), while medium questions show moderate gains (+4.9 points strict). The overall pattern suggests that LangExtract provides the most benefit when the question requires either direct fact retrieval (mostly easy questions) or complex relational reasoning (mostly hard questions), with intermediate utility for questions of moderate difficulty.

\subsection{Proprietary Baseline}

GPT-5.4 achieves 83.3\% strict and 95.0\% lenient accuracy on the baseline condition, substantially exceeding the best small model with augmentation (Qwen3 30B A3B at 71.7\% / 91.7\%). The 11.6-point strict accuracy gap between the best augmented small model and GPT-5.4 suggests that entity-aware preprocessing narrows but does not close the performance differential. The residual gap likely reflects both parameter-count advantages and broader training-data exposure to literary analysis patterns.

\section{Discussion}\label{sec:discuss}

\subsection{Mechanism: Why LangExtract Works}

LangExtract improves performance because it changes the \textit{representation} of the problem. Rather than appending token-level annotations to every row, it provides a semantically organized overview that the model can use as a scaffold for navigating the full prompt. The structure is aligned with the cognitive demands of the benchmark: questions frequently probe \textit{who knew what}, \textit{who was related to whom}, \textit{why a decision was taken}, and \textit{how one event precipitated another}. The overview functions as a narrative map, not merely a named-entity pass.

Two properties underpin this effectiveness. First, the overview abstracts without information loss---it summarizes entities and relationships while the full dataset remains available for detailed verification. Second, it achieves this with compact token cost: approximately 2,000 additional tokens on a $\sim$48,000-token prompt, a 4\% increase.

\subsection{Where LangExtract Helps Most}

The difficulty-stratified results confirm that LangExtract's largest absolute gains occur on hard questions. Consider Q3: ``Based on Benvolio's and Montague's observations before Romeo confesses anything, what symptoms are already documented, and what cause remains unresolved?'' Answering correctly requires identifying relevant scenes, extracting specific symptoms, recognizing that the cause is explicitly stated as unknown, and synthesizing these elements into a coherent response. The LangExtract overview aids scene location through character-indexed structure and surfaces themes of unexplained sorrow, while the model still must read and reason over the full dataset to extract and synthesize the precise symptoms.

\subsection{Model-Size Effects}

The results reveal a non-monotonic relationship between model size and benefit from augmentation. Ministral~3B and Mistral Small~3.2 show the largest improvements, while Qwen3 30B A3B---already strong at baseline---shows smaller gains. This pattern supports three inferences:

\begin{enumerate}
\item \textbf{Smaller models benefit more}: Models with fewer parameters possess less intrinsic capacity for narrative reconstruction, making external structure proportionally more valuable.
\item \textbf{Diminishing returns exist}: Once a model is sufficiently capable to handle long-context reasoning effectively (Qwen3 30B A3B at 70\% baseline), the marginal benefit of preprocessing decreases.
\item \textbf{The sweet spot is 3B--24B}: Models in this parameter range exhibit the most favorable balance between improvement magnitude and absolute performance.
\end{enumerate}

\subsection{Limitations}

Several limitations qualify the interpretation of these results.

\textbf{Context window constraints}: Llama~1B failed under LangExtract augmentation because the combined prompt ($\sim$60.3K tokens) exceeded the provider's 60K context limit. This is not an implementation error but an intrinsic cost of the method: preprocessing that improves accuracy for one model class may break compatibility for another.

\textbf{Full-dataset prompting}: The current benchmark sends the complete dataset for every question, conflating raw reasoning ability with long-context retention, tolerance for prompt expansion, and context-window capacity. A method that appears weak in this evaluation may perform well in a retrieval-augmented setup where question-specific evidence slices replace the full dataset.

\textbf{Extraction quality}: Gemini 2.5 Flash is not domain-adapted for Shakespearean language. While LLM-based extraction handles figurative language more robustly than traditional NER, it still produces some false or low-value extractions. Domain-specific fine-tuning or more targeted few-shot examples would likely improve extraction quality further.

\textbf{Verifier variance}: The verifier is itself a language model rather than a deterministic scoring script. While the constrained three-class output taxonomy mitigates this, some evaluator noise is unavoidable.

\section{Conclusion}

This work demonstrates that structured entity extraction preprocessing using LangExtract can measurably improve question-answering accuracy for small open-weight language models on a narrative comprehension benchmark. The principal findings are:

\begin{enumerate}
\item LangExtract consistently improves strict accuracy across all successful models (1B--30B), with mean improvement of +6.1 points (30.6\% $\rightarrow$ 36.7\%).
\item The largest gains occur on hard questions requiring relational and causal reasoning (+9.2 points strict; 26.9\% $\rightarrow$ 36.1\%), confirming that structured overviews are most valuable when the search burden is highest.
\item Compact, globally organized structure outperforms raw text alone, demonstrating that representational design matters as much as extraction quality.
\item Smaller models benefit proportionally more, with the 3B--24B range representing the sweet spot where external structure most effectively compensates for limited internal reasoning capacity.
\end{enumerate}

The pipeline establishes a practical approach for augmenting small language models on long-context narrative tasks. By prepending a structured overview before the full dataset, models can navigate complex narratives more efficiently without parameter-count increases or domain-specific fine-tuning.

\subsection{Future Work}

Several extensions would strengthen this line of research: (1)~question-specific evidence retrieval to reduce context pressure and improve small-model compatibility, (2)~domain-adapted extraction via fine-tuning on literary text, (3)~multi-work testing to assess generalizability beyond a single play, (4)~systematic ablation of extraction classes, passes, and buffer sizes, and (5)~hybrid approaches combining LangExtract's global overview with retrieval-based evidence selection.

\begin{thebibliography}{20}

\bibitem{b1} K.-J. Kim, C.-H. Lee, S.-E. Bae, J.-H. Choi, and W. Kang, ``Digital forensics in law enforcement: A case study of LLM-driven evidence analysis,'' \textit{Forensic Sci. Int.: Digit. Investig.}, vol.~54, p.~301939, 2025. DOI: \url{10.1016/j.fsidi.2025.301939}

\bibitem{b2} C. Grajeda, F. Breitinger, and I. Baggili, ``Availability of datasets for digital forensics---And what is missing,'' \textit{Digit. Investig.}, vol.~22, pp.~94--105, 2017. DOI: \url{10.1016/j.diin.2017.06.004}

\bibitem{b3} Z. Yin, Z. Wang, W. Xu, J. Zhuang, P. Mozumder, A. Smith, and W. Zhang, ``Digital forensics in the age of large language models,'' arXiv preprint arXiv:2504.02963v1, Apr. 2025. [Online]. Available: \url{https://arxiv.org/abs/2504.02963}

\bibitem{b4} H. Zhou, W. Xu, J. Dehlinger, S. Chakraborty, and L. Deng, ``Forensic intelligence graphs: An LLM approach to digital evidence extraction and relationship analysis,'' Towson Univ. \& Univ. of Baltimore, 2025.

\bibitem{b5} D. Bamman, O. Lewke, and A. Mansoor, ``An annotated dataset of literary entities,'' in \textit{Proc. NAACL-HLT}, 2019. [Online]. Available: \url{https://aclanthology.org/N19-1220/}

\bibitem{b6} J. Brooke, A. Tong, and G. Hirst, ``Bootstrapped text-level named entity recognition for literature,'' in \textit{Proc. ACL}, 2016. [Online]. Available: \url{https://aclanthology.org/P16-2056/}

\bibitem{b7} H. Vala, D. Jurgens, A. Piper, and S. Ray, ``Mr. Bennet, his coachman, and the Archbishop walk into a bar but only one of them gets recognized: On the difficulty of detecting characters in literary texts,'' in \textit{Proc. EMNLP}, 2015. [Online]. Available: \url{https://aclanthology.org/D15-1088/}

\bibitem{b8} Google, ``LangExtract: A Python library for structured information extraction from unstructured text,'' GitHub repository, 2025. [Online]. Available: \url{https://github.com/google/langextract} | DOI: \url{10.5281/zenodo.17015089}

\bibitem{b9} T. Ko\v{c}isk\'{y}, J. Schwarz, P. Blunsom, C. Dyer, G. Melis, N. Hermann, E. Grefenstette, and K.~M. Hermann, ``The NarrativeQA reading comprehension corpus,'' \textit{Trans. Assoc. Comput. Linguistics}, vol.~6, pp.~317--328, 2018. [Online]. Available: \url{https://arxiv.org/abs/1712.07040}

\bibitem{b10} T. Bonomo \textit{et al.}, ``LiteraryQA: Towards effective evaluation of long-document narrative QA,'' in \textit{Proc. EMNLP}, 2025. [Online]. Available: \url{https://aclanthology.org/2025.emnlp-main.1729/}

\bibitem{b11} S. Angelidis, M. Ballesteros, and J. Henderson, ``Book QA: Stories of challenges and opportunities,'' in \textit{Proc. MRQA Workshop at EMNLP}, 2019. [Online]. Available: \url{https://aclanthology.org/D19-5819/}

\bibitem{b12} U. Haddii, ``Shakespeare plays dialogues,'' Kaggle dataset, 2024. [Online]. Available: \url{https://www.kaggle.com/datasets/umerhaddii/shakespeare-plays-dialogues}

\bibitem{b13} G. Michel, J. Brooke, and A. Piper, ``Improving quotation attribution with fictional character embeddings,'' in \textit{Findings of EMNLP}, 2024. [Online]. Available: \url{https://aclanthology.org/2024.findings-emnlp.744/}

\bibitem{b14} F. Yang, M. Borowicz, and A. Piper, ``Character identification in literary texts,'' in \textit{Proc. AAAI}, 2022. [Online]. Available: \url{https://cdn.aaai.org/ojs/21709/21709-13-25722-1-2-20220628.pdf}

\bibitem{b15} R. Dufour, M. Ballesteros, and A. Piper, ``BERT meets d'Artagnan: Data augmentation for robust character detection in novels,'' in \textit{Proc. LREC}, 2022. [Online]. Available: \url{https://hal.univ-lorraine.fr/EC-NANTES/hal-03617722v1}

\bibitem{b16} J. Lee, C. Lim, B. Jin, M. Min, and H. Kim, ``DF-graph: Structured and explainable analysis of communication data for digital forensics,'' in \textit{Proc. DFRWS APAC}, Nov. 2025.

\bibitem{b17} W. Gale, K. Church, and D. Yarowsky, ``One sense per discourse,'' in \textit{Proc. DARPA Speech and Natural Language Workshop}, 1992. [Online]. Available: \url{https://aclanthology.org/H92-1045/}

\bibitem{b18} H.~T. Duong and T.~A. Nguyen-Thi, ``A review: Preprocessing techniques and data augmentation for sentiment analysis,'' \textit{Comput. Social Netw.}, vol.~8, no.~1, 2021. DOI: \url{10.1186/s40649-020-00080-x}

\bibitem{b19} J. Wei and K. Zou, ``EDA: Easy data augmentation techniques for boosting performance on text classification tasks,'' in \textit{Proc. ICLR}, 2019. [Online]. Available: \url{https://arxiv.org/abs/1901.11196}

\bibitem{b20} OpenRouter, ``OpenRouter API documentation.'' [Online]. Available: \url{https://openrouter.ai/docs}

\end{thebibliography}

\appendices

\section{Appendix A: Sample LangExtract Output}

Example structured overview generated by LangExtract (excerpt):

{\footnotesize
\begin{verbatim}
## Characters
- Chorus (1 mention)
- Sampson (12 mentions) -- servant of Capulet
- Gregory (10 mentions) -- servant of Capulet
- Abram (3 mentions) -- servant of Montague
- Benvolio (28 mentions) -- nephew of Montague,
  friend of Romeo
- Tybalt (18 mentions) -- nephew of
  Lady Capulet
...

## Relationships
- Sampson <-> Gregory: Co-servants, engage
  in banter about feud
- Benvolio <-> Romeo: Friends, Benvolio
  attempts to counsel Romeo
- Tybalt <-> Benvolio: Antagonists, Tybalt
  escalates conflict
...

## Key Events
- Act I, Scene I: Street brawl between
  Capulet/Montague servants
- Act I, Scene II: Capulet plans feast,
  Romeo learns of Rosaline
...
\end{verbatim}
}

\section{Appendix B: Complete Benchmark Questions}

The full set of 60 question-answer pairs used in the evaluation, organized by difficulty classification.

\subsection*{Easy Questions (Q2, Q4, Q11, Q15, Q23, Q32, Q40, Q41, Q46, Q52, Q58)}

\begin{description}
\item[Q2] What prior pattern of street violence does Prince Escalus cite when escalating the next penalty to death? \\
\textit{Answer:} Because their feud has already caused repeated civil brawls and endangered Verona's public peace.

\item[Q4] Which declared status of Rosaline explains why Romeo classifies her as unattainable? \\
\textit{Answer:} Because she has sworn chastity and refuses love, courtship, and seduction.

\item[Q11] What identity data does Juliet obtain from the Nurse about Romeo, and what conflict does that information immediately create? \\
\textit{Answer:} She learns that he is Romeo, a Montague, and the only son of her family's great enemy.

\item[Q15] Which type of oath does Juliet reject as unreliable evidence of constancy, and why? \\
\textit{Answer:} Because the moon is changeable, and she does not want his love to seem equally variable.

\item[Q23] Which witness account does Lady Capulet challenge as biased, and what relationship does she cite as grounds for distrust? \\
\textit{Answer:} Lady Capulet does, because Benvolio is related to the Montagues.

\item[Q32] What enforcement measures does Capulet threaten if Juliet refuses the marriage order? \\
\textit{Answer:} He threatens to drag her to church and, if she still refuses, cast her off to beg, starve, and die in the streets.

\item[Q40] Which wedding items does Capulet explicitly relabel as funeral items after Juliet is found dead? \\
\textit{Answer:} He says every wedding item turns into its funeral opposite: instruments become bells, feast becomes burial meal, hymns become dirges, and bridal flowers serve a corpse.

\item[Q41] Who delivers the first report that shapes Romeo's understanding of Juliet's condition in Act~V, and what corrective document is missing? \\
\textit{Answer:} Balthasar tells Romeo Juliet has been laid in the Capulet monument, but he has no letter from Friar Laurence.

\item[Q46] What observed behavior causes Balthasar to remain nearby instead of fully leaving? \\
\textit{Answer:} He hides nearby because Romeo looks dangerous and Balthasar distrusts his intentions.

\item[Q52] What exact quarantine conditions prevent Friar John from completing delivery of Friar Laurence's letter? \\
\textit{Answer:} He is quarantined with another friar in a suspected plague house, and no messenger will carry the letter for fear of infection.

\item[Q58] What physical evidence leads Juliet to infer poison, and why does she switch to the dagger? \\
\textit{Answer:} She sees the cup in Romeo's hand and concludes poison killed him, and when kissing him yields no poison for her, she takes his dagger and stabs herself.
\end{description}

\subsection*{Medium Questions (Q1, Q5--Q8, Q10, Q14, Q16, Q20--Q22, Q24--Q28, Q30, Q33, Q35, Q37, Q39, Q43--Q44, Q47--Q51, Q54--Q56)}

\begin{description}
\item[Q1] Which gesture does Sampson intentionally direct at the Montague servants, and what denial does he immediately use to limit legal exposure? \\
\textit{Answer:} He bites his thumb at the Montague servants, then tries to evade blame by saying he is biting his thumb, but not at them.

\item[Q5] When Paris seeks permission to marry Juliet, what age, waiting-period, and consent conditions does Capulet put on the request? \\
\textit{Answer:} Juliet is still too young, Paris should wait, and he must win Juliet's own consent because Capulet's approval is only part of the decision.

\item[Q6] What information chain links Romeo to the Capulet feast: who cannot read the guest list, who reads it, what name is found, and who uses that fact to influence Romeo? \\
\textit{Answer:} Capulet's illiterate servant asks him to read the guest list, Romeo learns Rosaline will be there, and Benvolio uses that to persuade him to attend.

\item[Q7] What level of commitment does Juliet give Lady Capulet regarding Paris, and what explicit limit does she place on her response? \\
\textit{Answer:} She says she will look to like him if looking leads to liking, but she will not let herself go further than her mother's consent allows.

\item[Q8] Before entering the feast, what risk forecast does Romeo make, and what principle leads him to proceed anyway? \\
\textit{Answer:} He fears the night will begin a star-governed chain that ends in untimely death, but he submits himself to providence and goes in.

\item[Q10] What hostile-family identity does Romeo learn about Juliet only after physical contact has already occurred? \\
\textit{Answer:} He discovers that she is a Capulet, so the woman he loves belongs to his enemy's house.

\item[Q14] What threat assessment does Juliet make about Romeo's presence in the orchard, and how does Romeo compare that threat with romantic rejection? \\
\textit{Answer:} She says her kinsmen will kill him if they find him there, and Romeo says her displeasure is more dangerous than their swords and that he would rather die than live without her love.

\item[Q16] What next-day verification process does Juliet require before treating Romeo's intentions as honorable? \\
\textit{Answer:} She asks him to send word through a messenger she will send, stating where and when they will be married, and if he is not honorable he should stop courting her.

\item[Q20] What secondhand evidence does the Nurse give Romeo about Juliet's attitude toward Paris? \\
\textit{Answer:} She tells him that Paris wants Juliet, but Juliet would rather look at a toad than at Paris and turns pale when Paris is praised.

\item[Q21] Which concealed kinship causes Romeo to de-escalate Tybalt's challenge, and when is that kinship later made explicit? \\
\textit{Answer:} Tybalt has become Romeo's kinsman through Romeo's secret marriage to Juliet, though Tybalt does not know it.

\item[Q22] What intervention by Romeo inadvertently creates the opening for Mercutio's fatal wound? \\
\textit{Answer:} Romeo rushes between the fighters to stop them, and Tybalt stabs under Romeo's arm.

\item[Q24] What causal factor leads the Prince to commute Romeo's expected death sentence to banishment? \\
\textit{Answer:} Because Tybalt had killed Mercutio first, so Romeo's killing of Tybalt is treated as retaliatory and softened from death to exile.

\item[Q25] Which reported development affects Juliet more severely than Tybalt's death? \\
\textit{Answer:} Romeo's banishment.

\item[Q26] What newly processed fact causes Juliet to reverse her first verbal attack on Romeo? \\
\textit{Answer:} She realizes Tybalt would have killed Romeo, so Tybalt's death means her husband lives.

\item[Q27] What evidence does Romeo cite to argue that banishment is functionally worse than death? \\
\textit{Answer:} Because exile means separation from Juliet and Verona, and even the lowest creatures may look on Juliet there while he cannot.

\item[Q28] Which three surviving advantages does Friar Laurence list when trying to counter Romeo's despair? \\
\textit{Answer:} Juliet is alive, Tybalt is dead instead of Romeo, and the law has changed Romeo's sentence from death to exile.

\item[Q30] Which incorrect diagnosis of Juliet's emotional state is shared by Lady Capulet and Paris, and how is it used to justify the marriage schedule? \\
\textit{Answer:} They think she is consumed by grief for Tybalt, and Paris says the wedding is being hurried to interrupt that grief.

\item[Q33] Which recommendation from the Nurse causes Juliet to treat her as no longer trustworthy? \\
\textit{Answer:} The Nurse tells Juliet to marry Paris because Romeo is banished and effectively useless to her, even belittling Romeo beside Paris.

\item[Q35] What demonstration by Juliet convinces Friar Laurence she can endure a deathlike deception? \\
\textit{Answer:} Her willingness to kill herself rather than marry Paris convinces him she has the nerve for a desperate deathlike stratagem.

\item[Q37] What room-access condition does Juliet create before taking the vial, and why is that condition necessary? \\
\textit{Answer:} She is following Friar Laurence's instructions so she can secretly take the vial without the Nurse in her chamber.

\item[Q39] When the Capulet household finds Juliet apparently dead, how does Friar Laurence attempt to reinterpret the event for them? \\
\textit{Answer:} He says heaven had a share in Juliet and now has her wholly, so she has been advanced to eternal life rather than merely lost.

\item[Q43] Which prior legal ruling and follow-up instruction explain Romeo's presence in Mantua at the start of Act~V? \\
\textit{Answer:} He is there because he was banished after Tybalt's death, and Friar Laurence told him to stay in Mantua until the marriage could be revealed and reconciliation attempted.

\item[Q44] What economic vulnerability allows Romeo to obtain poison from the Apothecary despite the law? \\
\textit{Answer:} His poverty overrules his will, and Romeo exploits that desperation by paying him.

\item[Q47] At the tomb, what role does Paris assign to himself and what support role does the Page perform? \\
\textit{Answer:} Paris comes to strew Juliet's grave with flowers and mourn her, while the Page stands watch, signals if anyone approaches, and later summons the watch.

\item[Q48] Which assumptions lead Paris to classify Romeo as a criminal intruder at the tomb? \\
\textit{Answer:} Paris thinks Romeo is the banished Montague who killed Tybalt, helped cause Juliet's death, and has now come to dishonor the dead.

\item[Q49] Before the duel with Paris, what does Romeo disclose about his mental state and self-directed intent? \\
\textit{Answer:} He says he is desperate, warns Paris not to provoke him, and admits he came armed against himself.

\item[Q50] After Paris is wounded, what new identity links does Romeo recognize, and how does that affect his treatment of Paris's body? \\
\textit{Answer:} Romeo recognizes Paris as Mercutio's kinsman and Juliet's would-be bridegroom, then grants Paris's wish to be laid in the tomb with Juliet.

\item[Q51] Which physical observations cause Romeo to doubt that Juliet has fully succumbed to death? \\
\textit{Answer:} Her lips and cheeks are still crimson rather than pale, so he thinks death has not yet erased her beauty.

\item[Q54] Which earlier family actions create the coercive environment that pushes Juliet toward the potion plan? \\
\textit{Answer:} Lady Capulet announces a Thursday marriage to Paris, and Capulet then forces the issue by threatening to drag Juliet to church or reject her if she refuses.

\item[Q55] Besides Friar Laurence, which named person is explicitly identified as knowing about the secret marriage? \\
\textit{Answer:} The Nurse.

\item[Q56] When Juliet regains consciousness in the tomb, what facts does she immediately know and what crucial deaths has she not yet learned? \\
\textit{Answer:} She remembers where she should be and immediately asks for Romeo, but she does not yet know that Romeo and Paris are dead.
\end{description}

\subsection*{Hard Questions (Q3, Q9, Q12--Q13, Q17--Q19, Q29, Q31, Q34, Q36, Q38, Q42, Q45, Q53, Q57, Q59--Q60)}

\begin{description}
\item[Q3] Based on Benvolio's and Montague's observations before Romeo confesses anything, what symptoms are already documented, and what cause remains unresolved? \\
\textit{Answer:} They know he wanders alone at dawn, weeps, sighs, avoids company, and shuts himself in darkness, but they still do not know the cause of his sorrow.

\item[Q9] By what identifying signal does Tybalt detect Romeo at the feast, and whose intervention prevents an immediate assault? \\
\textit{Answer:} Tybalt recognizes Romeo by his voice, but Capulet forbids violence because Romeo is reputed virtuous and a fight would disgrace the feast and disrupt the guests.

\item[Q12] In Juliet's balcony speech, which element of Romeo's identity does she explicitly identify as the obstacle to their relationship? \\
\textit{Answer:} She means why must you be Romeo, meaning why must he belong to the enemy name Montague, not where are you.

\item[Q13] Before Romeo reveals his presence, what contingency does Juliet state regarding Romeo's name and her own status as a Capulet? \\
\textit{Answer:} She has already said that either Romeo should renounce his name or she herself will cease to be a Capulet, because only his name, not his person, is her enemy.

\item[Q17] What conflict-resolution objective does Friar Laurence explicitly cite when deciding to assist the marriage? \\
\textit{Answer:} He agrees because he hopes their marriage can reconcile the two hostile households.

\item[Q18] Before Romeo returns in Act~II, Scene~IV, what outdated model of his emotional state do Mercutio and Benvolio still use? \\
\textit{Answer:} They still think Rosaline has made him love-sick, and Mercutio doubts that such a wounded Romeo can handle Tybalt's challenge.

\item[Q19] What warning does the Nurse issue about Romeo's intent, and what operational plan does Romeo return through her? \\
\textit{Answer:} She warns him not to deceive Juliet or lead her into a fool's paradise, and he tells her to bring Juliet to Friar Laurence's cell that afternoon for shrift and marriage and then wait behind the abbey wall for ladder-cords.

\item[Q29] Following the banishment ruling, what immediate movement plan and longer-term reconciliation plan does Friar Laurence assign to Romeo? \\
\textit{Answer:} Romeo should go to Juliet that night, leave before the watch, hide in Mantua, and wait while Friar Laurence works to reveal the marriage, reconcile the families, and seek a pardon.

\item[Q31] Which ambiguous statement allows Lady Capulet to infer that Juliet wants Romeo dead, and what does Juliet actually mean by it? \\
\textit{Answer:} She uses ambiguous phrasing so Lady Capulet hears a wish for Romeo's death, while Juliet means that her own heart is dead until she sees him.

\item[Q34] In the Friar's cell, which responses by Juliet sound compliant to Paris but avoid actual acceptance? \\
\textit{Answer:} She answers in equivocations: she says she may be a wife only when she may be one, says she loves him without naming Paris, and says even her face is not her own.

\item[Q36] What are the required steps, actors, and timing constraints in Friar Laurence's potion operation? \\
\textit{Answer:} She must appear cheerful and consent, sleep alone, drink the vial in bed, seem dead for forty-two hours, be laid in the Capulet vault, and then be recovered by Romeo and the Friar for flight to Mantua.

\item[Q38] What failure scenarios does Juliet model before consuming the potion? \\
\textit{Answer:} She fears it may fail, may actually be poison from the Friar, may wake her too early in the vault to suffocate, or may leave her mad among corpses, bones, and Tybalt's ghost.

\item[Q42] Which broken communication link causes Romeo to act on a false death report? \\
\textit{Answer:} Friar Laurence's letter explaining the potion never reaches Romeo because Friar John is quarantined, so Balthasar's report of Juliet's burial goes uncorrected.

\item[Q45] What cover story does Romeo give Balthasar for entering the tomb, and what later evidence shows it is not his full motive? \\
\textit{Answer:} He says he mainly needs a ring from Juliet's finger, but later evidence shows he came to die beside Juliet.

\item[Q53] What end-to-end recovery plan did Friar Laurence originally design for Juliet's waking, and which failure point prevents execution? \\
\textit{Answer:} Romeo was supposed to learn the plan by letter, come to the vault when Juliet awoke, and take her to Mantua, but the undelivered letter leaves him ignorant and the reunion never happens.

\item[Q57] What two factors cause Friar Laurence's extraction attempt to fail after Juliet wakes? \\
\textit{Answer:} He hears the watch coming and says their plan has been thwarted, and although he offers to hide Juliet in a nunnery, she refuses and he loses nerve at the approaching noise.

\item[Q59] In his formal explanation, how does Friar Laurence reconstruct the causal chain from the secret marriage to the final deaths? \\
\textit{Answer:} He says the secret marriage was followed by Tybalt's death and Romeo's banishment, Juliet then pined and was pushed toward Paris, she threatened suicide and took the sleeping potion, the letter failed, Romeo and Paris died before she awoke, and Juliet then killed herself.

\item[Q60] After Romeo's letter is read, what findings does the Prince make, and what remedial action do Capulet and Montague take? \\
\textit{Answer:} He concludes the letter confirms Friar Laurence's account and that the feud has brought this scourge, while he too is punished for tolerating it, and Capulet and Montague reconcile and promise gold statues for Juliet and Romeo.
\end{description}

\end{document}
