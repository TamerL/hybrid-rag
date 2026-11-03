# Document Chunking Strategies for RAG

Chunking is one of the highest-leverage and most under-discussed design decisions in RAG systems. Poor chunking produces either noisy contexts or incomplete evidence.

## Fixed-Size Chunking
Split every N characters or tokens with optional overlap. Simple and fast. Risks cutting sentences or logical units mid-way.

## Recursive Character Text Splitting
Used by LangChain's RecursiveCharacterTextSplitter. Tries a hierarchy of separators (paragraphs → sentences → words → characters) until chunks fit the size budget. Preserves larger semantic units when possible.

Typical settings for English technical text:
- chunk_size: 400–800 tokens
- chunk_overlap: 10–20% of chunk_size

## Semantic Chunking
Embed consecutive sentences or small windows and split when cosine similarity drops below a threshold. More expensive at indexing time but can produce more coherent chunks.

## Document Structure Aware
For Markdown, HTML, or PDFs with clear headings:
- Split on headings first
- Keep section title as metadata or prefix
- Respect code blocks, tables, and lists

## Practical Recommendations
1. Prefer recursive splitting with sentence-aware separators for general corpora.
2. Always store source document ID, section title, and character offsets as metadata.
3. Overlap is cheap insurance against boundary effects.
4. Measure retrieval metrics (Recall@k, nDCG) when changing chunk size; do not assume larger is better.
5. For long documents, consider hierarchical retrieval (summary → section → paragraph).

In this repository we use recursive character splitting with tiktoken length function, 512-token target size, and 64-token overlap. These values were chosen as a balanced starting point for technical documentation of moderate length.
