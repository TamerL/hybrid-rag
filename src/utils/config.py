"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[2]


class ChunkingConfig(BaseModel):
    strategy: str = "recursive_character"
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    length_function: str = "tiktoken"
    encoding_name: str = "cl100k_base"


class EmbeddingsConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 32


class SparseConfig(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class RetrievalConfig(BaseModel):
    top_k_dense: int = 20
    top_k_sparse: int = 20
    fusion_method: str = "rrf"
    rrf_k: int = 60
    top_k_hybrid: int = 15


class RerankingConfig(BaseModel):
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    device: str = "cpu"


class GenerationConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    include_citations: bool = True
    citation_style: str = "inline"


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )
    retrieval_metrics: list[str] = Field(
        default_factory=lambda: ["precision_at_k", "recall_at_k", "mrr"]
    )
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o-mini"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "Production Hybrid RAG API"
    version: str = "0.1.0"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    eval_dir: str = "data/eval"
    corpus_glob: str = "data/raw/**/*.{md,txt}"


class AppConfig(BaseModel):
    project: dict[str, Any] = Field(default_factory=dict)
    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config and return a validated AppConfig."""
    if path is None:
        path = ROOT_DIR / "configs" / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig.model_validate(raw)


class Settings(BaseSettings):
    """Environment-backed settings (API keys, etc.)."""

    openai_api_key: str | None = None
    openai_base_url: str | None = None  # for compatible endpoints
    huggingface_token: str | None = None
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()
