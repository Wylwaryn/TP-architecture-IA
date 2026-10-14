"""
Backend FastAPI (TD2) — recherche hiérarchique.

Flux :
  1. embedding de la requête (bge-m3 via Ollama)
  2. recherche vectorielle (knn) sur l'index "data_clusters"
       -> récupère le champ "data_index" du cluster le plus proche (ex. "data_1")
  3. recherche HYBRIDE (knn + BM25) dans cet index data_X          (TD2-6)
  4. reranking LLM par logprobs, en parallèle                      (TD3)
  5. tri par pertinence, renvoi des résultats
"""
import os

import httpx
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from rerank import rerank

ES_URL      = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
N_PROBE     = int(os.getenv("N_PROBE", "1"))
CLUSTERS_INDEX = "data_clusters"

app = FastAPI(title="IA Search — clusters + hybride + rerank")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

es = AsyncElasticsearch(ES_URL, request_timeout=60)


@app.on_event("shutdown")
async def _close():
    await es.close()


async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{OLLAMA_URL}/api/embed",
                              json={"model": EMBED_MODEL, "input": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embeddings"][0]


@app.get("/health")
async def health():
    try:
        ok = await es.ping()
    except Exception:
        ok = False
    return {"status": "ok", "elasticsearch": ok}


@app.get("/search")
async def search(
    q: str = Query(..., description="Requête de recherche"),
    probe: int = Query(N_PROBE, ge=1, le=20, description="Nb de clusters sondés"),
    candidates: int = Query(20, ge=1, le=100, description="Candidats avant rerank"),
    top_k: int = Query(10, ge=1, le=50, description="Nb de résultats renvoyés"),
):
    qvec = await embed(q)

    # 1. Recherche vectorielle sur data_clusters -> data_index du/des cluster(s)
    knn_res = await es.search(
        index=CLUSTERS_INDEX,
        knn={"field": "embedding", "query_vector": qvec,
             "k": probe, "num_candidates": 100},
        source=["data_index"], size=probe,
    )
    indices = [h["_source"]["data_index"] for h in knn_res["hits"]["hits"]]
    if not indices:
        return {"query": q, "indices": [], "results": []}

    # 2. Recherche HYBRIDE (knn + BM25) dans le(s) index du cluster
    hybrid_res = await es.search(
        index=",".join(indices),
        knn={"field": "embedding", "query_vector": qvec,
             "k": candidates, "num_candidates": 100},
        query={"match": {"line": q}},
        source=["line"], size=candidates,
    )
    hits = [
        {"index": h["_index"], "line": h["_source"]["line"],
         "search_score": round(h["_score"], 4)}
        for h in hybrid_res["hits"]["hits"]
    ]
    if not hits:
        return {"query": q, "indices": indices, "results": []}

    # 3. Rerank par logprobs (parallèle), tri décroissant
    reranked = await rerank(q, hits)

    return {"query": q, "indices": indices,
            "candidates": len(reranked), "results": reranked[:top_k]}
