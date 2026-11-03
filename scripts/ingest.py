#!/usr/bin/env python3
"""Ingest corpus and build indexes."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import typer
from rich import print as rprint

from src.pipeline import RAGPipeline
from src.utils.config import load_config
from src.utils.logging import setup_logging

app = typer.Typer(help="Ingest documents and build retrieval indexes")
logger = setup_logging(name="ingest")


@app.command()
def run(
    force: bool = typer.Option(False, "--force", help="Rebuild indexes even if they exist"),
    config: str = typer.Option(None, help="Path to config YAML"),
) -> None:
    """Load raw documents, chunk, and build dense + sparse indexes."""
    cfg = load_config(config)
    pipeline = RAGPipeline(config=cfg)
    pipeline.initialize(force_rebuild=force)
    stats = pipeline.get_stats()
    rprint("[green]Ingestion complete[/green]")
    rprint(stats)


if __name__ == "__main__":
    app()
