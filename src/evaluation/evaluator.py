"""Retrieval and RAGAS-style evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from langchain_core.documents import Document

from src.pipeline import RAGPipeline
from src.utils.config import AppConfig, ROOT_DIR
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


# ---------------------------------------------------------------------------
# Offline retrieval metrics (require relevance judgments)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved_ids[:k]
    return sum(1 for r in top if r in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    pipeline: RAGPipeline,
    eval_set: list[dict[str, Any]],
    k_values: list[int] = (1, 3, 5),
) -> dict[str, float]:
    """
    Compute Precision@k, Recall@k, MRR.

    Each item in eval_set:
      {
        "question": str,
        "relevant_chunk_ids": list[str]   # or relevant_file_ids
      }
    """
    metrics: dict[str, list[float]] = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics["mrr"] = []

    for item in eval_set:
        q = item["question"]
        relevant = set(item.get("relevant_chunk_ids") or item.get("relevant_file_ids") or [])
        docs = pipeline.retrieve(q)
        retrieved_ids = [
            d.metadata.get("chunk_id") or d.metadata.get("file_id") for d in docs
        ]

        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(retrieved_ids, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_ids, relevant, k))
        metrics["mrr"].append(mrr(retrieved_ids, relevant))

    summary = {name: float(np.mean(vals)) for name, vals in metrics.items()}
    return summary



# ---------------------------------------------------------------------------
# RAGAS end-to-end evaluation (supports local open-source judges)
# ---------------------------------------------------------------------------

def _build_judge_llm(config: AppConfig):
    """Build the LLM used by RAGAS as judge. Defaults to OpenAI (can be switched to Ollama)."""
    provider = getattr(config.evaluation, "judge_provider", "ollama") or "ollama"
    model = getattr(config.evaluation, "judge_model", "llama3.2") or "llama3.2"
    provider = provider.lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "langchain-ollama is required for local RAGAS judge.\n"
                "Install: pip install langchain-ollama\n"
                "And make sure Ollama is running with the model pulled."
            ) from e
        logger.info("RAGAS judge: Ollama model=%s", model)
        return ChatOllama(model=model, temperature=0.0)

    if provider == "openai":
        import os
        from langchain_openai import ChatOpenAI
        from src.utils.config import get_settings
        settings = get_settings()
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        logger.info("RAGAS judge: OpenAI model=%s", model)
        return ChatOpenAI(model=model, temperature=0.0, api_key=api_key)

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import pipeline as hf_pipeline
        logger.info("RAGAS judge: HuggingFace model=%s", model)
        pipe = hf_pipeline(
            "text-generation",
            model=model,
            device_map="auto",
            max_new_tokens=512,
            temperature=0.01,
            do_sample=False,
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    raise ValueError(f"Unsupported judge_provider: {provider}. Use ollama | openai | huggingface")


def run_ragas(
    pipeline: RAGPipeline,
    eval_set: list[dict[str, Any]],
    metrics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Run RAGAS evaluation using a (preferably local) LLM as judge.

    Each item in eval_set should have at least:
      - question
      - ground_truth (optional but improves some metrics)
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "ragas and datasets are required for end-to-end evaluation. "
            "Install with: pip install ragas datasets"
        ) from e

    # Build local (or configured) judge
    judge_llm = _build_judge_llm(pipeline.config)
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # Reuse the same embeddings the pipeline already uses
    ragas_embeddings = LangchainEmbeddingsWrapper(pipeline.embeddings)

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    selected = []
    for name in (metrics or list(metric_map.keys())):
        if name not in metric_map:
            continue
        m = metric_map[name]
        # Attach the local LLM / embeddings where needed
        if hasattr(m, "llm"):
            m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embeddings
        selected.append(m)

    records = []
    for item in eval_set:
        q = item["question"]
        result = pipeline.query(q)
        contexts = [c.page_content for c in result["contexts"]]
        records.append(
            {
                "question": q,
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": item.get("ground_truth", item.get("answer", "")),
            }
        )

    ds = Dataset.from_list(records)
    logger.info("Running RAGAS on %d examples with local/configured judge...", len(records))
    result = evaluate(
        ds,
        metrics=selected,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return dict(result)



def load_eval_set(path: Optional[Path] = None) -> list[dict[str, Any]]:
    if path is None:
        path = ROOT_DIR / "data" / "eval" / "golden_set.json"
    if not path.exists():
        logger.warning("Eval set not found at %s; returning empty list", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)
