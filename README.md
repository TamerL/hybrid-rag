# hybrid-rag

> Production-style hybrid RAG with dense + sparse retrieval, re-ranking, and citations

A clean, modular, production-oriented Retrieval-Augmented Generation (RAG) pipeline.

It demonstrates:

- Real (small but coherent) technical corpus
- Documented chunking strategy
- Hybrid retrieval (dense + sparse) with Reciprocal Rank Fusion
- Cross-encoder re-ranking
- Inline citations
- Offline retrieval metrics + RAGAS-style end-to-end evaluation
- FastAPI serving layer
- Docker support
- Honest analysis of failure modes and limitations

---

## Problem Statement

Vanilla LLM generation suffers from hallucination, stale knowledge, and lack of provenance. RAG grounds answers in a corpus, but naïve implementations (single dense retriever + fixed chunks) frequently fail on keyword-heavy queries, boundary effects, and ranking quality.

This project implements a **hybrid retrieval + re-ranking** stack that is closer to what teams ship in production, together with evaluation harnesses and explicit documentation of where it still breaks.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Documents  │────▶│   Chunker    │────▶│  Dense Index    │
│  (Markdown) │     │ (Recursive)  │     │  (FAISS)        │
└─────────────┘     └──────────────┘     └────────┬────────┘
                               │                  │
                               ▼                  │
                        ┌──────────────┐          │
                        │ Sparse Index │          │
                        │ (BM25)       │          │
                        └──────┬───────┘          │
                               │                  │
                               ▼                  ▼
                        ┌────────────────────────────┐
                        │   Reciprocal Rank Fusion   │
                        └─────────────┬──────────────┘
                                      │
                                      ▼
                        ┌────────────────────────────┐
                        │  Cross-Encoder Re-ranker   │
                        │  (ms-marco-MiniLM)         │
                        └─────────────┬──────────────┘
                                      │ top-k contexts
                                      ▼
                        ┌────────────────────────────┐
                        │     LLM Generator          │
                        │  (Ollama / HF / OpenAI)    │
                        │  + inline citations        │
                        └────────────────────────────┘
```

**Key design decisions**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunking | RecursiveCharacterTextSplitter, 512 tok, 64 overlap | Preserves semantic units better than naïve fixed-size; overlap reduces boundary loss |
| Dense | `all-MiniLM-L6-v2` + FAISS | Strong quality/size trade-off; fully local; no API cost for embeddings |
| Sparse | BM25 (via LangChain) | Complements dense on rare terms / exact matches |
| Fusion | Reciprocal Rank Fusion (k=60) | Parameter-light, robust, widely used in production hybrid systems |
| Re-ranking | Cross-encoder `ms-marco-MiniLM-L-6-v2` | Substantial precision lift on shortlists; applied only to top hybrid candidates |
| Generation | `gpt-4o-mini` (configurable) | Cheap, capable, good instruction following for citations |
| Serving | FastAPI | Lightweight, async-ready, easy to containerize |

---

## Tech Stack

- **Python 3.10+**
- **LangChain / LangGraph** (orchestration primitives)
- **FAISS** (dense ANN)
- **rank_bm25 / LangChain BM25Retriever** (sparse)
- **sentence-transformers** (embeddings + cross-encoder) — fully local
- **OpenAI** (default) or Ollama / HuggingFace for local generation
- **FastAPI + Uvicorn** (API)
- **RAGAS + datasets** (evaluation)
- **Pydantic / pydantic-settings** (config)
- **Docker** (optional deployment)

---

## Repository Structure

```
hybrid-rag/
├── configs/config.yaml          # All tunable hyperparameters
├── data/
│   ├── raw/                     # Source markdown corpus
│   ├── processed/               # FAISS index (generated)
│   └── eval/golden_set.json     # Small evaluation set
├── src/
│   ├── data/                    # Loader + chunker
│   ├── retrieval/               # Dense, sparse, hybrid, reranker
│   ├── generation/              # Prompts + LLM wrapper
│   ├── evaluation/              # Retrieval metrics + RAGAS
│   ├── api/                     # FastAPI app
│   ├── pipeline.py              # End-to-end orchestrator
│   └── utils/                   # Config, logging
├── scripts/
│   ├── ingest.py                # Build indexes
│   └── evaluate.py              # Metrics + demo queries
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick Start

### 1. Environment

```bash
git clone <your-repo-url> hybrid-rag
cd hybrid-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` → `.env` and set your OpenAI key:

```bash
cp .env.example .env
# then edit .env and add:
# OPENAI_API_KEY=sk-...
```

> Embeddings and the re-ranker are always local.  
> Generation + RAGAS judge default to OpenAI (`gpt-4o-mini`).  
> You can switch both to Ollama later if you want a fully free setup (see config).

### 2. Ingest & Index

```bash
python scripts/ingest.py
# or force rebuild
python scripts/ingest.py --force
```

### 3. Query from CLI

```bash
python scripts/evaluate.py demo "What is Reciprocal Rank Fusion?"
```

### 4. Start API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Optional: Fully local (Ollama) setup

If you prefer zero API cost:

1. Install Ollama and pull a model (`ollama pull llama3.2`)
2. In `configs/config.yaml` set:

```yaml
generation:
  provider: ollama
  model: "llama3.2"

evaluation:
  judge_provider: ollama
  judge_model: "llama3.2"
```

Note: smaller local models often produce less reliable RAGAS scores (JSON parsing errors are common).


## Evaluation

### Retrieval inspection

```bash
python scripts/evaluate.py retrieval
```

Prints top retrieved chunks for the first few golden questions so you can manually verify relevance and later add exact `relevant_chunk_ids` for quantitative Precision/Recall/MRR.

### RAGAS (end-to-end)

Defaults to OpenAI (`gpt-4o-mini`) as the judge for reliable scores.

```bash
python scripts/evaluate.py ragas
```

Metrics reported: Faithfulness, Answer Relevancy, Context Precision, Context Recall.

To use a local judge instead, change in `configs/config.yaml`:

```yaml
evaluation:
  judge_provider: ollama
  judge_model: "llama3.2"
```

**Note on the golden set**: The included set is intentionally small (8 questions). Expand it for more meaningful numbers.

## Key Metrics & Expected Behavior

On this small technical corpus you should observe:

- High retrieval of the correct source document for keyword and semantic questions about Transformers, BM25, RRF, chunking, and RAGAS metrics.
- Re-ranker typically surfaces the most precise chunk in the top-1 or top-2 position.
- Generation produces answers with `[1]`, `[2]` citations that map back to the returned sources.

Because the corpus is tiny, absolute Recall@5 is near 1.0 for well-formed questions. The interesting failure cases appear on:

- Ambiguous or out-of-scope questions
- Questions that require synthesis across many documents
- Very short or keyword-poor queries

---

## Failure Cases & Limitations (Honest Analysis)

1. **Tiny corpus**  
   The included documents are a convenience sample. Real production systems index thousands to millions of chunks; scaling introduces approximate search quality trade-offs and index management complexity not exercised here.

2. **No hierarchical / multi-vector retrieval**  
   Long documents are split flatly. Hierarchical (summary → section) or multi-vector (ColBERT-style) approaches often improve recall on complex queries.

3. **LLM-as-judge bias**  
   RAGAS metrics inherit the biases and failure modes of the judge model. Always validate a subset with humans.

4. **Re-ranker latency**  
   Cross-encoders are accurate but expensive. At higher QPS you would need distillation, approximate re-ranking, or cascading.

5. **Chunk boundary effects**  
   Even with overlap, critical definitions can still be split. Structure-aware or semantic chunking would help further.

6. **No query rewriting / HyDE / multi-query**  
   Single-query retrieval is used. Production systems frequently expand or rewrite the query.

7. **No access control or document-level security**  
   All chunks are equally visible.

8. **Evaluation set size**  
   8 examples are useful for smoke-testing, not for statistically robust claims. Expand before using numbers in a resume or paper.

These limitations are intentional: the repository prioritizes clarity, modularity, and teachable design decisions over maximal SOTA numbers.

---

## Model Card / Assumptions

| Component | Model / Method | Assumptions |
|-----------|----------------|-------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | English technical text; 384-dim; cosine; fully local |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Trained on MS MARCO; good for short passages; fully local |
| Generator (default) | **OpenAI `gpt-4o-mini`** | Requires `OPENAI_API_KEY`. Switch to Ollama for local use |
| Generator (alt) | HuggingFace local or OpenAI | See `configs/config.yaml` |
| Chunking | 512 / 64 recursive | Tuned for ~technical prose; retune for other domains |
| Fusion | RRF k=60 | Standard default; rarely needs retuning |

**Data assumptions**: Corpus is trusted, static, and English. No PII. No continuous update pipeline.

**Cost**: Default path uses OpenAI for generation + RAGAS judge (low cost with `gpt-4o-mini`). Embeddings and re-ranking stay local/free.

**Safety**: The system will refuse (via the prompt) to answer when context is insufficient, but it is not a hardened safety layer.

---

## Extending the Project

Ideas for further development:

- Swap FAISS for Qdrant / Weaviate / pgvector and add metadata filtering
- Add query rewriting (multi-query or HyDE) with LangGraph
- Implement a small continuous evaluation dashboard
- Distill the cross-encoder or replace with a bi-encoder late-interaction model
- Add structured output / tool-calling on top of the RAG answer
- Expand the golden set and publish retrieval + RAGAS numbers with confidence intervals

---

## License

MIT. Corpus files are original short technical summaries written for this repository (not copied from copyrighted papers).

---

## Acknowledgments

- Vaswani et al., “Attention Is All You Need”
- Robertson & Zaragoza, BM25
- Reimers & Gurevych, Sentence-BERT
- LangChain, RAGAS, FAISS, and sentence-transformers communities
