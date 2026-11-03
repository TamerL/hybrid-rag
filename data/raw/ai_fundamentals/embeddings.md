# Text Embeddings and Dense Retrieval

Dense embeddings map text into continuous vector spaces where semantic similarity corresponds to geometric proximity (usually cosine similarity or inner product).

## Sentence Transformers
The sentence-transformers library (Reimers & Gurevych, 2019) popularized bi-encoder architectures trained with contrastive objectives on NLI and paraphrase data. Models such as all-MiniLM-L6-v2 provide strong performance at low latency and small size (~80MB).

Key properties:
- Fixed-size vectors (e.g., 384 dimensions for MiniLM)
- Asymmetric vs symmetric retrieval: query and document may use different encoders
- Normalization often improves cosine-based ranking

## Training Objectives
Modern embedding models are trained with:
- Contrastive loss (InfoNCE)
- Multiple negatives ranking loss
- Hard negative mining
- Instruction tuning (e.g., E5, BGE, GTE families)

## Approximate Nearest Neighbor Search
Exact search is O(N). Production systems use:
- HNSW (Hierarchical Navigable Small World graphs) — used by FAISS, Chroma, Qdrant, Weaviate
- IVF (Inverted File Index) + Product Quantization
- DiskANN / ScaNN for very large scales

FAISS (Facebook AI Similarity Search) remains a standard library for research and many production deployments because of its mature C++ core and Python bindings.
