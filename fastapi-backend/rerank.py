"""
Reranking (TD3) — modèle LLM via Ollama, scoring par logprobs.

Méthode (conforme au TD3) :
  - on demande au modèle si le document est pertinent, réponse contrainte à "yes"/"no" ;
  - on récupère les logprobs du token de réponse ;
  - on somme les probabilités des variantes yes et no (yes/Yes/..., no/No/...) ;
  - score = p_yes / (p_yes + p_no) ;
  - tri décroissant.
Les documents sont scorés EN PARALLÈLE (asyncio.gather) : chaque paire (query, doc)
est une requête séparée, ce qui exploite mieux le KV-cache (cf. TD4).
"""
import os
import math
import asyncio

import httpx

OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.5:4b")

_SYSTEM = ("You judge whether the document is relevant to the query. "
           "Reply with a single word: yes or no.")


async def _score(client: httpx.AsyncClient, query: str, document: str) -> float:
    """Probabilité de pertinence p_yes / (p_yes + p_no) dans [0, 1]."""
    r = await client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": RERANK_MODEL,
            "think": False,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 2},
            "logprobs": True,
            "top_logprobs": 20,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Query: {query}\nDocument: {document}"},
            ],
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    # Token de réponse = 1er token généré ; on lit ses alternatives (top_logprobs).
    entries = data.get("logprobs") or []
    if not entries:
        return 0.0
    alts = entries[0].get("top_logprobs", [])

    p_yes = p_no = 0.0
    for alt in alts:
        t = alt["token"].strip().lower()
        p = math.exp(alt["logprob"])
        if t == "yes":
            p_yes += p
        elif t == "no":
            p_no += p

    return p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.0


async def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Score chaque candidat en parallèle, trie par pertinence décroissante."""
    async with httpx.AsyncClient() as client:
        scores = await asyncio.gather(
            *(_score(client, query, c["line"]) for c in candidates)
        )
    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(s, 4)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates
