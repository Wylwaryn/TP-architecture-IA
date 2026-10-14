"""
Script d'indexation — Partie "Script python d'indexation des données".

Pipeline (cf. schéma du projet) :
  1. Lire le dataset (1 intitulé de consultation par ligne).
  2. Calculer l'embedding de chaque intitulé via Ollama (modèle bge-m3, 1024 dims).
  3. Clusteriser les embeddings avec K-Means (k clusters).
  4. Indexer dans Elasticsearch :
       - un index `clusters`  = les CENTROÏDES des clusters (recherche vectorielle kNN) ;
       - un index `data_{k}` par cluster = les DOCUMENTS du cluster (recherche lexicale BM25).

Idée d'architecture : l'embedding ne sert qu'à ROUTER une requête vers le bon cluster
(kNN sur 5 centroïdes au lieu de 1000 documents). La recherche fine à l'intérieur du
cluster est faite en BM25 (lexical), puis un reranker LLM réordonne les candidats.
"""
import os
import sys
import time
import numpy as np
import requests
from sklearn.cluster import KMeans
from elasticsearch import Elasticsearch, helpers

# --------------------------------------------------------------------------- #
# Configuration (surchargée par les variables d'environnement dans Docker)
# --------------------------------------------------------------------------- #
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ES_URL      = os.getenv("ES_URL", "http://localhost:9200")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))
N_CLUSTERS  = int(os.getenv("N_CLUSTERS", "5"))
DATA_FILE   = os.getenv("DATA_FILE", "../data/objet_consultations_1000.txt")
BATCH_SIZE  = int(os.getenv("EMBED_BATCH", "32"))
CLUSTERS_INDEX = "clusters"


# --------------------------------------------------------------------------- #
# 1. Embeddings via Ollama
# --------------------------------------------------------------------------- #
def embed_batch(texts):
    """Encode une liste de textes -> liste de vecteurs (via /api/embed d'Ollama)."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_all(documents):
    vectors = []
    for start in range(0, len(documents), BATCH_SIZE):
        chunk = documents[start:start + BATCH_SIZE]
        vectors.extend(embed_batch(chunk))
        print(f"  embeddings {min(start + BATCH_SIZE, len(documents))}/{len(documents)}", flush=True)
    return np.array(vectors, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 2. Attente de la disponibilité des services (robustesse au démarrage Docker)
# --------------------------------------------------------------------------- #
def wait_for(url, name, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code < 500:
                print(f"[ok] {name} disponible", flush=True)
                return
        except requests.RequestException:
            pass
        print(f"[..] attente de {name} ...", flush=True)
        time.sleep(3)
    sys.exit(f"[err] {name} indisponible après {timeout}s ({url})")


# --------------------------------------------------------------------------- #
# 3. Création des index Elasticsearch
# --------------------------------------------------------------------------- #
def create_indices(es):
    # Index des centroïdes : recherche vectorielle (kNN cosinus).
    es.indices.delete(index=CLUSTERS_INDEX, ignore_unavailable=True)
    es.indices.create(
        index=CLUSTERS_INDEX,
        mappings={
            "properties": {
                "cluster_id": {"type": "integer"},
                "size":       {"type": "integer"},
                "centroid": {
                    "type": "dense_vector",
                    "dims": EMBED_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    )

    # Un index par cluster : recherche lexicale BM25 (analyseur français).
    for k in range(N_CLUSTERS):
        name = f"data_{k}"
        es.indices.delete(index=name, ignore_unavailable=True)
        es.indices.create(
            index=name,
            settings={"analysis": {"analyzer": {
                "default": {"type": "french"}  # stemming + stopwords + élision FR
            }}},
            mappings={
                "properties": {
                    "line_id":    {"type": "integer"},
                    "cluster_id": {"type": "integer"},
                    "line":       {"type": "text"},  # champ BM25
                }
            },
        )
    print(f"[ok] index créés : {CLUSTERS_INDEX} + data_0..data_{N_CLUSTERS - 1}", flush=True)


# --------------------------------------------------------------------------- #
# 4. Indexation en masse (Bulk API)
# --------------------------------------------------------------------------- #
def bulk_index(es, documents, labels, centroids):
    # Centroïdes
    centroid_docs = [
        {
            "_index": CLUSTERS_INDEX,
            "_id": k,
            "_source": {
                "cluster_id": k,
                "size": int((labels == k).sum()),
                "centroid": centroids[k].tolist(),
            },
        }
        for k in range(N_CLUSTERS)
    ]
    helpers.bulk(es, centroid_docs)

    # Documents, routés vers data_{cluster}
    doc_actions = (
        {
            "_index": f"data_{int(labels[i])}",
            "_id": i,
            "_source": {
                "line_id": i,
                "cluster_id": int(labels[i]),
                "line": documents[i],
            },
        }
        for i in range(len(documents))
    )
    helpers.bulk(es, doc_actions)

    for k in range(N_CLUSTERS):
        es.indices.refresh(index=f"data_{k}")
    es.indices.refresh(index=CLUSTERS_INDEX)
    print(f"[ok] {len(documents)} documents indexés", flush=True)


# --------------------------------------------------------------------------- #
def main():
    # Attendre ES et Ollama
    wait_for(f"{ES_URL}/", "Elasticsearch")
    wait_for(f"{OLLAMA_URL}/api/version", "Ollama")

    # Lecture du dataset
    with open(DATA_FILE, encoding="utf-8") as f:
        documents = [ln.strip() for ln in f if ln.strip()]
    print(f"[ok] {len(documents)} documents chargés depuis {DATA_FILE}", flush=True)

    # Embeddings
    print(f"[..] calcul des embeddings ({EMBED_MODEL}, dim={EMBED_DIM})", flush=True)
    embeddings = embed_all(documents)
    assert embeddings.shape[1] == EMBED_DIM, \
        f"dimension embedding {embeddings.shape[1]} != EMBED_DIM {EMBED_DIM}"

    # Clustering
    print(f"[..] K-Means (k={N_CLUSTERS})", flush=True)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    for k in range(N_CLUSTERS):
        print(f"    cluster {k}: {(labels == k).sum()} documents", flush=True)

    # Indexation
    es = Elasticsearch(ES_URL, request_timeout=60)
    create_indices(es)
    bulk_index(es, documents, labels, centroids)
    print("[done] indexation terminée.", flush=True)


if __name__ == "__main__":
    main()
