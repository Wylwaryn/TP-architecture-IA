"""
Script d'indexation (TD1-3).

Pipeline :
  lines            = get_data()                      # lit le dataset
  line_embeddings  = get_data_embeddings(lines)      # embedding via Ollama
  cluster_to_lines = cluster_data(line_embeddings)   # K-Means
  main()           -> indexe dans Elasticsearch :
      - un index par cluster : data_1, data_2, ...   { "embedding": [...], "line": "..." }
      - un index "data_clusters"                     { "embedding": [...], "data_index": "data_1" }
"""
import os
import sys
import time

import numpy as np
import requests
from sklearn.cluster import KMeans
from elasticsearch import Elasticsearch, helpers

OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ES_URL      = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))
N_CLUSTERS  = int(os.getenv("N_CLUSTERS", "5"))
DATA_FILE   = os.getenv("DATA_FILE", "../data/objet_consultations_1000.txt")
BATCH_SIZE  = int(os.getenv("EMBED_BATCH", "32"))
CLUSTERS_INDEX = "data_clusters"


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


def get_data():
    """Lit le dataset (un intitulé de consultation par ligne)."""
    with open(DATA_FILE, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    print(f"[ok] {len(lines)} documents chargés depuis {DATA_FILE}", flush=True)
    return lines


def get_data_embeddings(lines):
    """Calcule les embeddings en les envoyant au docker Ollama."""
    print(f"[..] embeddings ({EMBED_MODEL}, dim={EMBED_DIM})", flush=True)
    vectors = []
    for start in range(0, len(lines), BATCH_SIZE):
        chunk = lines[start:start + BATCH_SIZE]
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": chunk},
            timeout=600,
        )
        r.raise_for_status()
        vectors.extend(r.json()["embeddings"])
        print(f"    {min(start + BATCH_SIZE, len(lines))}/{len(lines)}", flush=True)
    embeddings = np.array(vectors, dtype=np.float32)
    assert embeddings.shape[1] == EMBED_DIM, \
        f"dim {embeddings.shape[1]} != EMBED_DIM {EMBED_DIM}"
    return embeddings


def cluster_data(line_embeddings):
    """K-Means : retourne (labels, centroïdes)."""
    print(f"[..] K-Means (k={N_CLUSTERS})", flush=True)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(line_embeddings)
    for k in range(N_CLUSTERS):
        print(f"    cluster {k+1}: {(labels == k).sum()} documents", flush=True)
    return labels, kmeans.cluster_centers_.astype(np.float32)


def data_index_name(label):
    """cluster 0-based -> nom d'index 1-based (data_1, data_2, ...)."""
    return f"data_{label + 1}"


def create_indices(es):
    # Un index par cluster : embedding (knn) + line (BM25 analyseur FR).
    for k in range(N_CLUSTERS):
        name = data_index_name(k)
        es.indices.delete(index=name, ignore_unavailable=True)
        es.indices.create(
            index=name,
            settings={"analysis": {"analyzer": {"default": {"type": "french"}}}},
            mappings={"properties": {
                "line": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": EMBED_DIM,
                              "index": True, "similarity": "cosine"},
            }},
        )
    # Index des clusters : centroïde + nom de l'index correspondant.
    es.indices.delete(index=CLUSTERS_INDEX, ignore_unavailable=True)
    es.indices.create(
        index=CLUSTERS_INDEX,
        mappings={"properties": {
            "data_index": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": EMBED_DIM,
                          "index": True, "similarity": "cosine"},
        }},
    )
    print(f"[ok] index créés : {CLUSTERS_INDEX} + data_1..data_{N_CLUSTERS}", flush=True)


def main():
    wait_for(f"{ES_URL}/", "Elasticsearch")
    wait_for(f"{OLLAMA_URL}/api/version", "Ollama")

    lines = get_data()
    line_embeddings = get_data_embeddings(lines)
    labels, centroids = cluster_data(line_embeddings)

    es = Elasticsearch(ES_URL, request_timeout=60)
    create_indices(es)

    # Documents -> index de leur cluster { embedding, line }
    docs = (
        {"_index": data_index_name(int(labels[i])), "_id": i,
         "_source": {"line": lines[i], "embedding": line_embeddings[i].tolist()}}
        for i in range(len(lines))
    )
    helpers.bulk(es, docs)

    # Clusters -> data_clusters { embedding, data_index }
    cluster_docs = [
        {"_index": CLUSTERS_INDEX, "_id": k,
         "_source": {"data_index": data_index_name(k),
                     "embedding": centroids[k].tolist()}}
        for k in range(N_CLUSTERS)
    ]
    helpers.bulk(es, cluster_docs)

    es.indices.refresh(index=f"{CLUSTERS_INDEX},data_*")
    print(f"[done] {len(lines)} documents indexés", flush=True)


if __name__ == "__main__":
    main()
