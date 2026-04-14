import asyncio
import logging
from typing import List, Dict

from config import RETRIEVAL_TOP_K
from retriever import retrieve, build_prompt, ask_ollama_async

log = logging.getLogger(__name__)

PERSONAS = {
    "analytical-critical": (
        "You are an analytical-critical evaluator. Break down the core arguments of the context passages, "
        "highlighting logic and structure. Use ONLY the provided context and cite passages (e.g. [1])."
    ),
    "adversarial-skeptical": (
        "You are an adversarial-skeptical debater. Identify biases, missing information, or weak points in "
        "the provided context. Use ONLY the provided context and cite passages."
    ),
    "creative-lateral": (
        "You are a creative-lateral thinker. Draw unexpected connections or innovative ideas from the provided "
        "context without inventing new facts. Use ONLY the provided context and cite passages."
    ),
    "historical-contextual": (
        "You are a historical-contextual historian. Place the core events or facts of the context into a "
        "broader timeline or historical trend. Use ONLY the provided context and cite passages."
    ),
    "economical-financial": (
        "You are an economical-financial analyst. Focus on market implications, funding, costs, and value "
        "creation mentioned in the context. Use ONLY the provided context and cite passages."
    ),
    "ethical-societal": (
        "You are an ethical-societal observer. Analyze the impact on people, communities, and moral "
        "frameworks based on the context. Use ONLY the provided context and cite passages."
    ),
    "technical-scientific": (
        "You are a technical-scientific expert. Focus on the mechanics, data accuracy, and technical "
        "specifications described in the context. Use ONLY the provided context and cite passages."
    ),
    "practical-pragmatic": (
        "You are a practical-pragmatic strategist. Identify the most immediate real-world applications "
        "and actionable steps suggested by the context. Use ONLY the provided context and cite passages."
    )
}

async def answer_multi_persona_async(query: str, top_k: int = RETRIEVAL_TOP_K) -> Dict:
    """
    Pulls top-k chunks for the query, builds 8 independent prompts based on the personas,
    and executes them concurrently via async Ollama requests.
    """
    # 1. Retrieve the shared context once
    chunks = retrieve(query, top_k=top_k)
    
    if not chunks:
        # Provide fallback for all 8 personas
        empty_responses = [
            {"persona": name, "answer": "No relevant context found in our index for this query."}
            for name in PERSONAS.keys()
        ]
        return {
            "query": query,
            "responses": empty_responses,
            "sources": []
        }

    sources = [
        {
            "title":  c.get("title", ""),
            "url":    c.get("url", ""),
            "source": c.get("source", ""),
            "score":  round(c.get("score", 0.0), 4),
        }
        for c in chunks
    ]

    # 2. Define the async task for a single persona
    async def fetch_persona(name: str, sys_prompt: str) -> Dict:
        prompt = build_prompt(query, chunks, system_override=sys_prompt)
        try:
            answer_text = await ask_ollama_async(prompt)
        except Exception as e:
            log.error(f"Persona '{name}' failed: {e}")
            answer_text = f"Error generating response: {e}"
        return {"persona": name, "answer": answer_text}

    # 3. Spin up all 8 prompts concurrently
    log.info(f"Dispatching query to {len(PERSONAS)} parallel personas...")
    tasks = [fetch_persona(name, sys_prompt) for name, sys_prompt in PERSONAS.items()]
    persona_results = await asyncio.gather(*tasks)

    return {
        "query": query,
        "responses": persona_results,
        "sources": sources
    }
