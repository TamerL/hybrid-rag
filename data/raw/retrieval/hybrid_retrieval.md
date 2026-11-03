# Hybrid Retrieval: Combining Dense and Sparse Signals

Pure dense retrieval can fail on rare terms, exact identifiers, or highly specific jargon. Pure sparse retrieval (BM25) cannot capture paraphrases or semantic equivalence. Hybrid retrieval aims to get the best of both.

## Sparse Retrieval (BM25)
BM25 is a probabilistic ranking function based on the Okapi model. It scores a document D for query Q as a sum over query terms of:

score(D, Q) = Σ IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))

where:
- f(q_i, D) is term frequency
- |D| is document length
- avgdl is average document length
- k1 and b are free parameters (commonly 1.2–2.0 and 0.75)

BM25 is extremely strong on keyword matching and remains a competitive baseline more than 30 years after introduction.

## Dense Retrieval
Dense retrievers embed queries and documents into the same vector space and rank by similarity. They excel when the query and relevant documents share meaning but not surface form.

## Fusion Methods
Two common approaches:

1. **Score Normalization + Weighted Sum**
   Normalize BM25 and dense scores to [0,1] then combine with a weight α.

2. **Reciprocal Rank Fusion (RRF)**
   RRF score = Σ 1 / (k + rank_i)
   where rank_i is the rank of the document in the i-th ranked list.
   RRF is parameter-light, robust, and often outperforms weighted sums without careful calibration.

## Re-ranking
After retrieving a broader candidate set (e.g., top-20 from hybrid), a cross-encoder re-ranker scores (query, document) pairs jointly. Cross-encoders are more accurate but slower; they are typically applied only to the shortlist.

Popular cross-encoders include ms-marco-MiniLM and the larger BGE-reranker series.
