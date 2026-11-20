# Label review sheet (for a human)

The relevance and answerability labels were produced by AI raters (see `judged_by` in `labels.json` / `questions.json`).
Only the items below were disputed or are debatable; everything else was unanimous. To change a decision, edit the `relevant`
list of that query in `labels.json` (arXiv ids), then run `python -m cli eval retrieval --collection <id> --labels data/eval/labels.json`.

Votes are `[first pass, rater A, rater B]` (1 = relevant). **Final** is the majority.

## Query 1: how does retrieval augmentation reduce hallucination

- `2501.09292` **To Retrieve or Not to Retrieve? Uncertainty Detection for Dynamic Retrieval Augmented Generation** — votes [1, 0, 0], final **not relevant**  
  Retrieval-Augmented Generation equips large language models with the capability to retrieve external knowledge, thereby mitigating hallucinations by incorporating information beyond the model's intrinsic abilities. However, most prior works have focused on invoking retrieval deterministically, which makes it unsuitable for tasks such as long-form question answering. Instead, dynamically performing retrieval by invoki…
  Your call: [ ] relevant  [ ] not relevant

- `2510.22344` **FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation** — votes [1, 0, 1], final **relevant**  
  While Retrieval-Augmented Generation (RAG) mitigates hallucination and knowledge staleness in Large Language Models (LLMs), existing frameworks often falter on complex, multi-hop queries that require synthesizing information from disparate sources. Current advanced RAG methods, employing iterative or adaptive strategies, lack a robust mechanism to systematically identify and fill evidence gaps, often propagating nois…
  Your call: [ ] relevant  [ ] not relevant

- `2309.15217` **Ragas: Automated Evaluation of Retrieval Augmented Generation** — votes [1, 0, 0], final **not relevant**  
  We introduce Ragas (Retrieval Augmented Generation Assessment), a framework for reference-free evaluation of Retrieval Augmented Generation (RAG) pipelines. RAG systems are composed of a retrieval and an LLM based generation module, and provide LLMs with knowledge from a reference textual database, which enables them to act as a natural language layer between a user and textual databases, reducing the risk of halluci…
  Your call: [ ] relevant  [ ] not relevant

- `2506.00054` **Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers** — votes [1, 1, 0], final **relevant**  
  Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to enhance large language models (LLMs) by conditioning generation on external evidence retrieved at inference time. While RAG addresses critical limitations of parametric knowledge storage-such as factual inconsistency and domain inflexibility-it introduces new challenges in retrieval quality, grounding fidelity, pipeline efficiency, and robustn…
  Your call: [ ] relevant  [ ] not relevant

- `2602.07739` **HypRAG: Hyperbolic Dense Retrieval for Retrieval Augmented Generation** — votes [0, 1, 0], final **not relevant**  
  Embedding geometry plays a fundamental role in retrieval quality, yet dense retrievers for retrieval-augmented generation (RAG) remain largely confined to Euclidean space. However, natural language exhibits hierarchical structure from broad topics to specific entities that Euclidean embeddings fail to preserve, causing semantically distant documents to appear spuriously similar and increasing hallucination risk. To a…
  Your call: [ ] relevant  [ ] not relevant

## Query 2: evaluating retrieval augmented generation systems

- `2506.00054` **Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers** — votes [0, 1, 0], final **not relevant**  
  Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to enhance large language models (LLMs) by conditioning generation on external evidence retrieved at inference time. While RAG addresses critical limitations of parametric knowledge storage-such as factual inconsistency and domain inflexibility-it introduces new challenges in retrieval quality, grounding fidelity, pipeline efficiency, and robustn…
  Your call: [ ] relevant  [ ] not relevant

- `2604.19899` **A Reproducibility Study of Metacognitive Retrieval-Augmented Generation** — votes [1, 0, 0], final **not relevant**  
  Recently, Retrieval Augmented Generation (RAG) has shifted focus to multi-retrieval approaches to tackle complex tasks such as multi-hop question answering. However, these systems struggle to decide when to stop searching once enough information has been gathered. To address this, \citet{zhou2024metacognitive} introduced Metacognitive Retrieval Augmented Generation (MetaRAG), a framework inspired by metacognition tha…
  Your call: [ ] relevant  [ ] not relevant

- `2603.16169` **Open-Source Reproduction and Explainability Analysis of Corrective Retrieval Augmented Generation** — votes [1, 0, 0], final **not relevant**  
  Corrective Retrieval Augmented Generation (CRAG) improves the robustness of RAG systems by evaluating retrieved document quality and triggering corrective actions. However, the original implementation relies on proprietary components including the Google Search API and closed model weights, limiting reproducibility. In this work, we present a fully open-source reproduction of CRAG, replacing proprietary web search wi…
  Your call: [ ] relevant  [ ] not relevant

- `2607.24767` **The Effect of Text Chunk Size on Retrieval-Augmented Generation Performance** — votes [1, 0, 0], final **not relevant**  
  Retrieval-Augmented Generation (RAG) systems have emerged as a powerful process for allowing large language models (LLMs) to retrieve relevant information to use as source material during text generation. A critical yet under-explored component of these systems is the granularity at which source documents are segmented into retrievable chunks. The size of these chunks has the potential to significantly influence gene…
  Your call: [ ] relevant  [ ] not relevant

- `2504.19754` **Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation** — votes [1, 0, 0], final **not relevant**  
  Retrieval-augmented generation (RAG) has become a transformative approach for enhancing large language models (LLMs) by grounding their outputs in external knowledge sources. Yet, a critical question persists: how can vast volumes of external knowledge be managed effectively within the input constraints of LLMs? Traditional methods address this by chunking external documents into smaller, fixed-size segments. While t…
  Your call: [ ] relevant  [ ] not relevant

- `2607.01852` **Evaluating Chunking Strategies for Retrieval-Augmented Generation on Academic Texts** — votes [1, 1, 0], final **relevant**  
  Retrieval-Augmented Generation (RAG) systems use the question-answering capabilities of Large Language Models (LLMs) to access information outside their parameters. We evaluate if cluster-based semantic chunking improves retrieval and answer quality compared to fixed-size and recursive chunking evaluating on long, structured academic theses using the Retrieval Augmented Generation Assessment (RAGAs) framework. RAGAs …
  Your call: [ ] relevant  [ ] not relevant

## Query 4: hybrid sparse and dense retrieval

- `2609.31660` **Parser, Chunking, and Embedding Interactions in Retrieval-Augmented Generation over Indian Government Regulatory Documents** — votes [0, 0, 1], final **not relevant**  
  Retrieval-augmented generation (RAG) pipelines are typically assembled from independently-chosen components -- a document parser, a chunking strategy, and an embedding model -- yet these choices are rarely evaluated jointly, and evaluations that do combine them are usually run on a single document or corpus. We present a controlled factorial study of 3 parsers, 3 chunking strategies, and 5 dense embedding models, tog…
  Your call: [ ] relevant  [ ] not relevant

- `2604.22849` **R$^3$AG: Retriever Routing for Retrieval-Augmented Generation** — votes [1, 0, 0], final **not relevant**  
  Retrieval-augmented generation (RAG) has become a cornerstone for knowledge-intensive tasks. However, the efficacy of RAG is often bottlenecked by the ``one-size-fits-all'' retrieval paradigm, as different queries exhibit distinct preferences for different retrievers. While recent routing techniques attempt to select the optimal retriever dynamically, they typically operate under a ``single and static capability'' as…
  Your call: [ ] relevant  [ ] not relevant

## Query 8: citations and attribution in generated answers

- `2603.09891` **Overview of the TREC 2025 Retrieval Augmented Generation (RAG) Track** — votes [0, 1, 1], final **relevant**  
  The second edition of the TREC Retrieval Augmented Generation (RAG) Track advances research on systems that integrate retrieval and generation to address complex, real-world information needs. Building on the foundation of the inaugural 2024 track, this year's challenge introduces long, multi-sentence narrative queries to better reflect the deep search task with the growing demand for reasoning-driven responses. Part…
  Your call: [ ] relevant  [ ] not relevant

- `2609.29075` **CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars** — votes [0, 1, 1], final **relevant**  
  Cancer registrars, including Oncology Data Specialists (ODSs), must interpret complex and frequently updated coding and staging standards. We developed CRISS (Cancer Registry Intelligent Support System), a retrieval-augmented generation (RAG) conversational assistant that provides rapid, citation-supported access to registry guidance. This study evaluated whether CRISS could (1) support accurate and citation-supporte…
  Your call: [ ] relevant  [ ] not relevant

## Query 9: domain specific retrieval augmented generation for healthcare

- `2512.00367` **Breaking It Down: Domain-Aware Semantic Segmentation for Retrieval Augmented Generation** — votes [0, 1, 1], final **relevant**  
  Document chunking is a crucial component of Retrieval-Augmented Generation (RAG), as it directly affects the retrieval of relevant and precise context. Conventional fixed-length and recursive splitters often produce arbitrary, incoherent segments that fail to preserve semantic structure. Although semantic chunking has gained traction, its influence on generation quality remains underexplored. This paper introduces tw…
  Your call: [ ] relevant  [ ] not relevant

## Query 10: reranking retrieved passages

- `2504.05731` **Retrieval Augmented Generation with Collaborative Filtering for Personalized Text Generation** — votes [0, 1, 1], final **relevant**  
  Recently, the personalization of Large Language Models (LLMs) to generate content that aligns with individual user preferences has garnered widespread attention. Personalized Retrieval-Augmented Generation (RAG), which retrieves relevant documents from the user's history to reflect their preferences and enhance LLM generation, is one commonly used approach for personalization. However, existing personalized RAG metho…
  Your call: [ ] relevant  [ ] not relevant

- `2608.08994` **Guardian Crawler: Retrieval-First Knowledge Discovery with Bounded LLM Augmentation for Noisy Web Intelligence** — votes [0, 1, 1], final **relevant**  
  Retrieving relevant evidence from noisy web data is challenging, particularly in sensitive domains containing incomplete reports, heterogeneous language, and irrelevant content. We present Guardian Crawler, a reproducible retrieval-first testbed for controlled experiments on knowledge discovery and evidence-grounded summarization over synthetic web-like corpora. The architecture combines BM25 retrieval with risk-awar…
  Your call: [ ] relevant  [ ] not relevant

- `2604.19899` **A Reproducibility Study of Metacognitive Retrieval-Augmented Generation** — votes [0, 1, 1], final **relevant**  
  Recently, Retrieval Augmented Generation (RAG) has shifted focus to multi-retrieval approaches to tackle complex tasks such as multi-hop question answering. However, these systems struggle to decide when to stop searching once enough information has been gathered. To address this, \citet{zhou2024metacognitive} introduced Metacognitive Retrieval Augmented Generation (MetaRAG), a framework inspired by metacognition tha…
  Your call: [ ] relevant  [ ] not relevant

- `2411.00744` **CARROT: A Learned Cost-Constrained Retrieval Optimization System for RAG** — votes [1, 0, 1], final **relevant**  
  Large Language Models (LLMs) have demonstrated impressive ability in generation and reasoning tasks but struggle with handling up-to-date knowledge, leading to inaccuracies or hallucinations. Retrieval-Augmented Generation (RAG) mitigates this by retrieving and incorporating external knowledge into input prompts. In particular, due to LLMs' context window limitations and long-context hallucinations, only the most rel…
  Your call: [ ] relevant  [ ] not relevant

## Query 11: knowledge graph retrieval augmented generation

- `2310.13848` **FABULA: Intelligence Report Generation Using Retrieval-Augmented Narrative Construction** — votes [0, 1, 1], final **relevant**  
  Narrative construction is the process of representing disparate event information into a logical plot structure that models an end to end story. Intelligence analysis is an example of a domain that can benefit tremendously from narrative construction techniques, particularly in aiding analysts during the largely manual and costly process of synthesizing event information into comprehensive intelligence reports. Manua…
  Your call: [ ] relevant  [ ] not relevant

- `2509.01030` **Identifying Origins of Place Names via Retrieval Augmented Generation** — votes [0, 1, 1], final **relevant**  
  Who is the "Batman" behind "Batman Street" in Melbourne? Understanding the historical, cultural, and societal narratives behind place names can reveal the rich context that has shaped a community. Although place names serve as essential spatial references in gazetteers, they often lack information about place name origins. Enriching these place names in today's gazetteers is a time-consuming, manual process that requ…
  Your call: [ ] relevant  [ ] not relevant

## Query 12: when should a model decide to retrieve

- `2404.14464` **Tree of Reviews: A Tree-based Dynamic Iterative Retrieval Framework for Multi-hop Question Answering** — votes [0, 1, 1], final **relevant**  
  Multi-hop question answering is a knowledge-intensive complex problem. Large Language Models (LLMs) use their Chain of Thoughts (CoT) capability to reason complex problems step by step, and retrieval-augmentation can effectively alleviate factual errors caused by outdated and unknown knowledge in LLMs. Recent works have introduced retrieval-augmentation in the CoT reasoning to solve multi-hop question answering. Howe…
  Your call: [ ] relevant  [ ] not relevant

- `2510.22344` **FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation** — votes [0, 0, 1], final **not relevant**  
  While Retrieval-Augmented Generation (RAG) mitigates hallucination and knowledge staleness in Large Language Models (LLMs), existing frameworks often falter on complex, multi-hop queries that require synthesizing information from disparate sources. Current advanced RAG methods, employing iterative or adaptive strategies, lack a robust mechanism to systematically identify and fill evidence gaps, often propagating nois…
  Your call: [ ] relevant  [ ] not relevant

## Query 13: query rewriting for retrieval

- `2501.14342` **Chain-of-Retrieval Augmented Generation** — votes [0, 1, 1], final **relevant**  
  This paper introduces an approach for training o1-like RAG models that retrieve and reason over relevant information step by step before generating the final answer. Conventional RAG methods usually perform a single retrieval step before the generation process, which limits their effectiveness in addressing complex queries due to imperfect retrieval results. In contrast, our proposed method, CoRAG (Chain-of-Retrieval…
  Your call: [ ] relevant  [ ] not relevant

- `2507.06956` **Investigating the Robustness of Retrieval-Augmented Generation at the Query Level** — votes [0, 0, 1], final **not relevant**  
  Large language models (LLMs) are very costly and inefficient to update with new information. To address this limitation, retrieval-augmented generation (RAG) has been proposed as a solution that dynamically incorporates external knowledge during inference, improving factual consistency and reducing hallucinations. Despite its promise, RAG systems face practical challenges-most notably, a strong dependence on the qual…
  Your call: [ ] relevant  [ ] not relevant

## Query 15: conversational search with retrieval

- `2409.18575` **Corpus-informed Retrieval Augmented Generation of Clarifying Questions** — votes [1, 0, 1], final **relevant**  
  This study aims to develop models that generate corpus informed clarifying questions for web search, in a way that ensures the questions align with the available information in the retrieval corpus. We demonstrate the effectiveness of Retrieval Augmented Language Models (RAG) in this process, emphasising their ability to (i) jointly model the user query and retrieval corpus to pinpoint the uncertainty and ask for cla…
  Your call: [ ] relevant  [ ] not relevant

## Borderline questions (labelled unanswerable; answerable from full text in a narrower setting)

- **How does reinforcement learning from human feedback train a reward model?** — the abstracts do not cover it, but the full text of one paper (RAG-Reward) describes reward modelling for RAG; the system answers it scoped to that setting.  Your call: [ ] keep unanswerable  [ ] mark answerable

- **How do transformers implement multi-head self-attention?** — the abstracts do not cover it, but the full text of a video-retrieval paper describes multi-head attention in its model; the system answers it scoped to that paper.  Your call: [ ] keep unanswerable  [ ] mark answerable
