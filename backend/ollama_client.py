"""Client Ollama : embedding (bge-m3) et reranking (LLM pointwise, qwen3.5:4b).

Note technique importante :
  Ollama n'expose PAS d'endpoint de reranking natif (feature request #3368 toujours
  ouverte). Les modèles "reranker" cross-encoder (type bge-reranker) ont une tête de
  classification qu'Ollama ne sait pas servir, et les GGUF "qwen3-reranker" renvoient
  n'importe quoi via /api/chat. On implémente donc un RERANK PAR LLM POINTWISE : on
  demande à un modèle instruct (qwen3.5:4b) un score de pertinence 0..10 pour chaque
  paire (requête, document). Technique reconnue (RankGPT pointwise), fiable et locale.
"""
import os
import re
import httpx

OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.5:4b")

_RERANK_SYSTEM = (
    "Tu es un juge de pertinence. Donne un score entier de 0 (totalement hors sujet) "
    "à 10 (parfaitement pertinent) indiquant à quel point le DOCUMENT répond à la "
    "REQUÊTE. Réponds UNIQUEMENT par l'entier, rien d'autre."
)


async def embed_query(client: httpx.AsyncClient, text: str) -> list[float]:
    """Embedding d'une requête (même modèle que l'indexation)."""
    r = await client.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embeddings"][0]


async def rerank_score(client: httpx.AsyncClient, query: str, document: str) -> float:
    """Score de pertinence normalisé dans [0, 1] pour une paire (requête, document)."""
    r = await client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": RERANK_MODEL,
            "think": False,            # pas de raisonnement : réponse directe
            "stream": False,
            "options": {"temperature": 0, "num_predict": 6},
            "messages": [
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user",
                 "content": f"REQUÊTE: {query}\nDOCUMENT: {document}\nScore:"},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "")
    m = re.search(r"\d+", content)
    score = int(m.group()) if m else 0
    return max(0.0, min(10.0, score)) / 10.0
