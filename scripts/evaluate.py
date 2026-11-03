#!/usr/bin/env python3
"""Run retrieval and/or RAGAS evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import typer
from rich import print as rprint
from rich.table import Table

from src.evaluation.evaluator import evaluate_retrieval, load_eval_set, run_ragas
from src.pipeline import RAGPipeline
from src.utils.config import load_config
from src.utils.logging import setup_logging

app = typer.Typer(help="Evaluate the RAG system")
logger = setup_logging(name="evaluate")


@app.command()
def retrieval(
    config: str = typer.Option(None, help="Path to config YAML"),
) -> None:
    """Compute offline retrieval metrics on the golden set."""
    cfg = load_config(config)
    pipeline = RAGPipeline(config=cfg)
    pipeline.initialize()

    eval_set = load_eval_set()
    if not eval_set:
        rprint("[yellow]No golden set found. Add data/eval/golden_set.json[/yellow]")
        raise typer.Exit(1)

    # For demo purposes the golden set does not yet contain exact chunk_ids.
    # In a real project you would label relevant chunk_ids after first ingestion.
    # Here we run retrieval and print the top results for manual inspection.
    rprint("[bold]Retrieval inspection (manual relevance labeling recommended)[/bold]\n")
    for item in eval_set[:3]:
        docs = pipeline.retrieve(item["question"])
        rprint(f"[cyan]Q:[/cyan] {item['question']}")
        for i, d in enumerate(docs[:3], 1):
            rprint(
                f"  [{i}] {d.metadata.get('title')} "
                f"(chunk={d.metadata.get('chunk_id')}, "
                f"score={d.metadata.get('rerank_score', d.metadata.get('rrf_score')):.3f})"
            )
        rprint()


@app.command()
def ragas(
    config: str = typer.Option(None, help="Path to config YAML"),
) -> None:
    """Run RAGAS end-to-end metrics (uses local Ollama judge by default)."""
    cfg = load_config(config)
    pipeline = RAGPipeline(config=cfg)
    pipeline.initialize()

    eval_set = load_eval_set()
    if not eval_set:
        rprint("[yellow]No golden set found.[/yellow]")
        raise typer.Exit(1)

    try:
        results = run_ragas(pipeline, eval_set, metrics=cfg.evaluation.metrics)
    except Exception as e:
        rprint(f"[red]RAGAS failed: {e}[/red]")
        rprint("Make sure Ollama is running with the judge model pulled, and ragas is installed.")
        raise typer.Exit(1)

    table = Table(title="RAGAS Results")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            table.add_row(k, f"{v:.4f}")
    rprint(table)


@app.command()
def demo(
    question: str = typer.Argument(..., help="Question to ask"),
    config: str = typer.Option(None),
) -> None:
    """Quick interactive query for debugging."""
    cfg = load_config(config)
    pipeline = RAGPipeline(config=cfg)
    pipeline.initialize()
    result = pipeline.query(question)
    rprint(f"\n[bold]Answer:[/bold]\n{result['answer']}\n")
    rprint("[bold]Citations:[/bold]")
    for c in result["citations"]:
        rprint(f"  [{c['id']}] {c['title']} ({c['source']})")


if __name__ == "__main__":
    app()
