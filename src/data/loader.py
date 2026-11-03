"""Document loading from local corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document

from src.utils.config import ROOT_DIR, AppConfig
from src.utils.logging import setup_logging

logger = setup_logging(name=__name__)


def _file_id(path: Path) -> str:
    """Stable short ID based on relative path."""
    rel = str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)
    return hashlib.sha1(rel.encode()).hexdigest()[:12]


def load_documents(config: AppConfig) -> list[Document]:
    """
    Load all markdown/text files under data/raw.

    Each Document receives metadata:
      - source: relative path
      - file_id: short hash
      - title: first H1 or filename
    """
    raw_dir = ROOT_DIR / config.data.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    docs: list[Document] = []
    patterns = ["**/*.md", "**/*.txt"]

    for pattern in patterns:
        for path in sorted(raw_dir.glob(pattern)):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                logger.warning("Skipping empty file: %s", path)
                continue

            # Extract a simple title
            title = path.stem.replace("_", " ").title()
            for line in text.splitlines()[:5]:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            rel_source = str(path.relative_to(ROOT_DIR))
            metadata = {
                "source": rel_source,
                "file_id": _file_id(path),
                "title": title,
                "filename": path.name,
            }
            docs.append(Document(page_content=text, metadata=metadata))
            logger.info("Loaded %s (%d chars)", rel_source, len(text))

    logger.info("Total documents loaded: %d", len(docs))
    return docs


def iter_documents(config: AppConfig) -> Iterator[Document]:
    """Generator version for large corpora."""
    yield from load_documents(config)
