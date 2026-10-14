"""
Backend FastAPI — moteur de recherche hiérarchique.

Flux de recherche (cf. schéma du projet) :
  1. Embedding de la requête (bge-m3 via Ollama).
  2. kNN vectoriel sur l'index `clusters` -> cluster(s) le(s) plus proche(s).
  3. Recherche BM25 (lexicale) dans le/les index `data_{cluster}` correspondant(s).
  4. Reranking LLM pointwise (qwen3.5:4b via Ollama), EN PARALLÈLE (asyncio.gather).
  5. Tri par score de rerank, renvoi des top_k résultats.
"""
import os
import asyncio

import httpx
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from ollama_client import embed_query, rerank_score

ES_URL     = os.getenv("ES_URL", "http://localhost:9200")
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "5"))
N_PROBE    = int(os.getenv("N_PROBE", "1"))          # nb de clusters sondés par défaut
CLUSTERS_INDEX = "clusters"

app = FastAPI(title="IA Search — recherche hiérarchique (embedding + BM25 + rerank)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

es = AsyncElasticsearch(ES_URL, request_timeout=60)


@app.on_event("shutdown")
async def _close():
    await es.close()


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
    probe: int = Query(N_PROBE, ge=1, le=N_CLUSTERS, description="Nb de clusters sondés"),
    candidates: int = Query(20, ge=1, le=100, description="Candidats BM25 avant rerank"),
    top_k: int = Query(10, ge=1, le=50, description="Nb de résultats renvoyés"),
):
    async with httpx.AsyncClient() as client:
        # 1. Embedding de la requête
        qvec = await embed_query(client, q)

        # 2. kNN sur les centroïdes -> clusters les plus proches
        knn_res = await es.search(
            index=CLUSTERS_INDEX,
            knn={"field": "centroid", "query_vector": qvec,
                 "k": probe, "num_candidates": N_CLUSTERS},
            source=["cluster_id"], size=probe,
        )
        clusters = [h["_source"]["cluster_id"] for h in knn_res["hits"]["hits"]]
        if not clusters:
            return {"query": q, "clusters": [], "results": []}

        # 3. Recherche BM25 dans le(s) index data_{cluster}
        target = ",".join(f"data_{c}" for c in clusters)
        bm25_res = await es.search(
            index=target,
            query={"match": {"line": q}},
            size=candidates,
        )
        hits = [
            {
                "line_id": h["_source"]["line_id"],
                "line": h["_source"]["line"],
                "cluster_id": h["_source"]["cluster_id"],
                "bm25_score": round(h["_score"], 4),
            }
            for h in bm25_res["hits"]["hits"]
        ]
        if not hits:
            return {"query": q, "clusters": clusters, "results": []}

        # 4. Reranking LLM pointwise, en parallèle
        scores = await asyncio.gather(
            *(rerank_score(client, q, h["line"]) for h in hits)
        )

    for h, s in zip(hits, scores):
        h["rerank_score"] = round(s, 4)

    # 5. Tri : rerank décroissant, BM25 comme départage
    hits.sort(key=lambda h: (h["rerank_score"], h["bm25_score"]), reverse=True)

    return {
        "query": q,
        "clusters": clusters,
        "candidates": len(hits),
        "results": hits[:top_k],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
