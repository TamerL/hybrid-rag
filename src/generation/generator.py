"""Answer generation with optional citations.

Supports fully open-source / free setups:
  - ollama          (recommended free local option)
  - huggingface     (local transformers pipeline)
  - openai          (optional, paid)
  - mock            (no model needed)
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.generation.prompts import SYSTEM_PROMPT, USER_TEMPLATE, format_context
from src.utils.config import AppConfig, get_settings
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


def build_llm(config: AppConfig) -> BaseChatModel:
    """Build the chat model according to config + environment.

    Priority for free/open-source usage:
      1. Set generation.provider: ollama  (install Ollama + pull a model)
      2. Set generation.provider: huggingface  (pure local, slower)
      3. provider: mock  (no LLM at all, useful for testing retrieval)
    """
    settings = get_settings()
    cfg = config.generation
    provider = (cfg.provider or "ollama").lower()

    # ---------- Fully free / open-source options ----------
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "langchain-ollama is required for the Ollama provider.\n"
                "Install with:  pip install langchain-ollama\n"
                "Also install Ollama from https://ollama.com and run e.g.:\n"
                "  ollama pull llama3.2"
            ) from e

        model = cfg.model or "llama3.2"
        logger.info("Using Ollama model: %s", model)
        return ChatOllama(
            model=model,
            temperature=cfg.temperature,
            num_predict=cfg.max_tokens,
        )

    if provider == "huggingface":
        # Pure local HuggingFace pipeline (no external service)
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            from transformers import pipeline as hf_pipeline
        except ImportError as e:
            raise ImportError(
                "transformers + langchain-huggingface required for local HF generation.\n"
                "pip install transformers accelerate langchain-huggingface"
            ) from e

        model_id = cfg.model or "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        logger.info("Loading local HuggingFace model: %s (this may take a while)...", model_id)

        pipe = hf_pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            do_sample=cfg.temperature > 0,
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    if provider == "mock":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(
            responses=[
                "This is a mock answer grounded in the retrieved context [1]. "
                "In a real deployment an LLM would synthesize a precise response."
            ]
        )

    # ---------- Optional paid / hosted ----------
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set. Either set the key or switch to a free provider:\n"
                "  generation.provider: ollama   (recommended)\n"
                "  generation.provider: huggingface\n"
                "  generation.provider: mock"
            )
        return ChatOpenAI(
            model=cfg.model or "gpt-4o-mini",
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=api_key,
            base_url=settings.openai_base_url,
        )

    raise ValueError(
        f"Unsupported generation provider: {provider}. "
        "Choose one of: ollama | huggingface | openai | mock"
    )


def generate_answer(
    question: str,
    contexts: list[Document],
    llm: BaseChatModel,
    config: AppConfig,
) -> dict[str, Any]:
    """
    Generate an answer conditioned on retrieved contexts.

    Returns a dict with:
      - answer: str
      - contexts: the documents used
      - citations: list of source metadata
    """
    if not contexts:
        return {
            "answer": "I could not find relevant information in the knowledge base to answer this question.",
            "contexts": [],
            "citations": [],
        }

    context_str = format_context(contexts)
    user_msg = USER_TEMPLATE.format(context=context_str, question=question)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    citations = []
    for i, doc in enumerate(contexts, start=1):
        citations.append(
            {
                "id": i,
                "title": doc.metadata.get("title"),
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "rerank_score": doc.metadata.get("rerank_score"),
                "rrf_score": doc.metadata.get("rrf_score"),
            }
        )

    return {
        "answer": answer,
        "contexts": contexts,
        "citations": citations,
    }
