"""FastAPI application exposing the RAG pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline import RAGPipeline
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)

# Global pipeline instance (initialized at startup)
pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    config = load_config()
    pipeline = RAGPipeline(config=config)
    pipeline.initialize()
    logger.info("API ready")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    config = load_config()
    app = FastAPI(
        title=config.api.title,
        version=config.api.version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    include_contexts: bool = False


class Citation(BaseModel):
    id: int
    title: Optional[str] = None
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    contexts: Optional[list[dict[str, Any]]] = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict[str, Any]:
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    return pipeline.get_stats()


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    try:
        result = pipeline.query(req.question)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(500, f"Generation failed: {e}") from e

    contexts_out = None
    if req.include_contexts:
        contexts_out = [
            {
                "content": c.page_content[:500],
                "metadata": c.metadata,
            }
            for c in result["contexts"]
        ]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        contexts=contexts_out,
    )


# For `uvicorn src.api.main:app`
if __name__ == "__main__":
    import uvicorn
    from src.utils.config import load_config

    cfg = load_config()
    uvicorn.run(
        "src.api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
    )
