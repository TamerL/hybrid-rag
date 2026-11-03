"""Prompt templates for generation with citations."""

SYSTEM_PROMPT = """You are a careful technical assistant that answers questions using only the provided context.
Rules:
1. Base every factual claim on the context. If the context is insufficient, say so clearly.
2. Use inline citations of the form [1], [2], ... that refer to the numbered sources below.
3. Prefer concise, precise answers. Use bullet points or short paragraphs when helpful.
4. Do not invent information that is not supported by the context.
"""

USER_TEMPLATE = """Context sources:
{context}

Question: {question}

Answer (with inline citations):"""


def format_context(docs: list, max_chars_per_doc: int = 1200) -> str:
    """Format retrieved documents as numbered sources for the prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Untitled")
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."
        parts.append(f"[{i}] Title: {title}\nSource: {source}\n{content}")
    return "\n\n".join(parts)
