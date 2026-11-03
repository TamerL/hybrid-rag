# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines a parametric language model with a non-parametric memory (usually a document index). At inference time the system retrieves relevant passages and conditions the generator on them.

## Motivation
Large language models have parametric knowledge that is:
- Frozen after training
- Prone to hallucination
- Difficult to update without expensive fine-tuning
- Opaque regarding sources

RAG mitigates these by grounding generation in retrieved evidence and enabling easy knowledge updates via the index.

## Canonical Pipeline
1. **Indexing**: Chunk documents → embed → store in vector database (+ optional sparse index)
2. **Retrieval**: Given a user query, retrieve top-k relevant chunks
3. **Augmentation**: Construct a prompt that includes the retrieved chunks and the original question
4. **Generation**: LLM produces an answer, ideally citing the supporting chunks

## Common Failure Modes
- Retrieval misses the relevant document (low recall)
- Retrieved documents are only tangentially related (low precision)
- Generator ignores the context or hallucinates beyond it
- Chunk boundaries split critical information
- Query-document vocabulary mismatch

## Evaluation
Modern evaluation goes beyond exact match. RAGAS and similar frameworks measure:
- Faithfulness (does the answer stay grounded in the context?)
- Answer Relevancy
- Context Precision / Context Recall

Human evaluation and domain-specific metrics remain essential for high-stakes applications.
