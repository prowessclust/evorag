"""
retriever.py — EvoRAG Phase 3: Retrieval Pipeline

Public API
----------
    retrieve(query, top_k)    → List[dict]   (top-k chunk dicts with 'score')
    build_prompt(query, chunks) → str        (assembled RAG prompt)
    ask_ollama(prompt)        → str          (LLM answer string)
    answer(query, top_k)      → dict         ({answer, sources, query})

This module is imported by api.py (Phase 3) and by the persona engine (Phase 4).
"""

import logging
from typing import List, Dict, Optional

import httpx

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RETRIEVAL_TOP_K,
)
from embedder import query as faiss_query, load_index, build_or_load

log = logging.getLogger(__name__)

# ── Module-level index cache (loaded once at import / first call) ─────────────
_index = None
_metadata = None


def _ensure_index():
    """Load the FAISS index from disk exactly once per process."""
    global _index, _metadata
    if _index is None or _metadata is None:
        _index, _metadata = load_index()


# ══════════════════════════════════════════════════════════════════════════════
# 1. RETRIEVE — semantic search over FAISS index
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
    """
    Embed `query` and return the top-k most similar chunks from the FAISS index.

    Args:
        query:  The user question / search string.
        top_k:  Number of chunks to return.

    Returns:
        List of chunk dicts (each has 'text', 'title', 'source', 'url', 'score', …).
    """
    _ensure_index()
    results = faiss_query(query, top_k=top_k, index=_index, metadata=_metadata)
    log.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}'")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROMPT ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(query: str, chunks: List[Dict], system_override: str = None) -> str:
    """
    Assemble a RAG prompt from the retrieved chunks.

    Format:
        System instruction
        Numbered context passages (title + source + text)
        User question

    Args:
        query:  The original user question.
        chunks: List of retrieved chunk dicts (must have 'text', 'title', 'source').
        system_override: Optional persona string to replace the default instruction.

    Returns:
        The full prompt string to send to Ollama.
    """
    system = system_override if system_override else (
        "You are a factual news analysis assistant. "
        "Use ONLY the context passages provided below to answer the question. "
        "Cite the passage number (e.g. [1], [2]) when you use information from it. "
        "If the context does not contain enough information to answer, say so clearly — "
        "do not invent facts."
    )

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        title  = chunk.get("title", "Untitled")[:100]
        source = chunk.get("source", "unknown")
        text   = chunk.get("text", "").strip()
        block  = (
            f"[{i}] Source: {source} | Title: {title}\n"
            f"{text}"
        )
        context_blocks.append(block)

    context_section = "\n\n".join(context_blocks)

    prompt = (
        f"{system}\n\n"
        f"--- CONTEXT ---\n"
        f"{context_section}\n"
        f"--- END CONTEXT ---\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# 3. OLLAMA CALL — synchronous HTTP (used by validate_phase3 & CLI)
# ══════════════════════════════════════════════════════════════════════════════

def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Send a prompt to Ollama's /api/generate endpoint (non-streaming).

    Args:
        prompt: The full prompt string.
        model:  Ollama model tag (default from config).

    Returns:
        The model's response text string.

    Raises:
        httpx.HTTPError or ConnectionError if Ollama is unreachable.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    log.info(f"Calling Ollama ({model}) — prompt length: {len(prompt)} chars")
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer_text = data.get("response", "").strip()
            log.info(f"Ollama responded ({len(answer_text)} chars).")
            return answer_text
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. ASYNC OLLAMA CALL — used by FastAPI endpoint (non-blocking)
# ══════════════════════════════════════════════════════════════════════════════

async def ask_ollama_async(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Async version of ask_ollama — used inside FastAPI route handlers.

    Args:
        prompt: The full prompt string.
        model:  Ollama model tag.

    Returns:
        The model's response text string.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    log.info(f"[async] Calling Ollama ({model}) — prompt length: {len(prompt)} chars")
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer_text = data.get("response", "").strip()
            log.info(f"[async] Ollama responded ({len(answer_text)} chars).")
            return answer_text
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. END-TO-END — retrieve + prompt + answer (sync, for CLI / tests)
# ══════════════════════════════════════════════════════════════════════════════

def answer(query: str, top_k: int = RETRIEVAL_TOP_K) -> Dict:
    """
    Full RAG pipeline — synchronous version.

    Steps:
        1. Retrieve top-k chunks from FAISS
        2. Build RAG prompt
        3. Send to Ollama
        4. Return structured result

    Returns:
        {
            "query":   str,
            "answer":  str,
            "sources": [{"title", "url", "source", "score"}, …]
        }
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "query":   query,
            "answer":  "No relevant context found in the index. Try fetching more articles first.",
            "sources": [],
        }

    prompt = build_prompt(query, chunks)
    answer_text = ask_ollama(prompt)

    sources = [
        {
            "title":  c.get("title", ""),
            "url":    c.get("url", ""),
            "source": c.get("source", ""),
            "score":  round(c.get("score", 0.0), 4),
        }
        for c in chunks
    ]

    return {
        "query":   query,
        "answer":  answer_text,
        "sources": sources,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. END-TO-END — async version (used by FastAPI)
# ══════════════════════════════════════════════════════════════════════════════

async def answer_async(query: str, top_k: int = RETRIEVAL_TOP_K) -> Dict:
    """
    Full RAG pipeline — async version used by the FastAPI endpoint.
    Identical logic to answer() but calls ask_ollama_async().
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "query":   query,
            "answer":  "No relevant context found in the index. Try fetching more articles first.",
            "sources": [],
        }

    prompt = build_prompt(query, chunks)
    answer_text = await ask_ollama_async(prompt)

    sources = [
        {
            "title":  c.get("title", ""),
            "url":    c.get("url", ""),
            "source": c.get("source", ""),
            "score":  round(c.get("score", 0.0), 4),
        }
        for c in chunks
    ]

    return {
        "query":   query,
        "answer":  answer_text,
        "sources": sources,
    }
