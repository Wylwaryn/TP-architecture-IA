# IA Search — Moteur de recherche de consultations

Projet d'évaluation « Architectures IA ». Moteur de recherche sémantique sur un corpus
de **1000 intitulés de marchés publics**, avec une architecture de **recherche hiérarchique** :
embedding → routage par cluster → BM25 → reranking.

## Architecture

```
                 ┌──────────── INDEXATION (one-shot) ────────────┐
 dataset ──▶ Ollama (bge-m3) ──▶ embeddings ──▶ K-Means ──▶ Elasticsearch
                                                              ├─ index "clusters"  (centroïdes, dense_vector)
                                                              └─ index "data_0..4" (documents, BM25 FR)

                 ┌──────────────── RECHERCHE ────────────────────┐
 React ──▶ FastAPI ──▶ 1. embed requête (bge-m3)
                       2. kNN sur "clusters"      → cluster le plus proche
                       3. BM25 sur "data_{k}"     → candidats
                       4. rerank LLM (qwen3.5:4b) → scores de pertinence  (parallèle)
                       5. tri + top_k             → React
```

**Pourquoi ce design ?** L'embedding ne sert qu'à **router** la requête vers le bon
cluster (kNN sur 5 centroïdes, pas sur 1000 vecteurs → ça passe à l'échelle, principe
des index IVF de FAISS). La recherche fine est **lexicale (BM25)**, excellente sur les
termes exacts, puis le **reranker** (cross-encodeur sémantique) corrige les angles morts
du lexical en relisant chaque paire (requête, document) ensemble.

## Composants

| Service | Rôle | Techno |
|---|---|---|
| `elasticsearch` | Base vectorielle + BM25 | Elasticsearch 8.12 |
| `ollama` + `ollama-init` | Modèles IA (embedding + rerank) | Ollama |
| `indexer` | Script Python d'indexation | Python, scikit-learn |
| `backend` | API de recherche | FastAPI |
| `frontend` | Interface de recherche | React (Vite) + nginx |

### Choix des modèles

- **Embedding : `bge-m3`** (1024 dims, multilingue, très bon en français). Le mapping
  Elasticsearch `dense_vector` est donc en `dims: 1024`, `similarity: cosine`.
- **Rerank : `qwen3.5:4b` en *pointwise LLM reranking***. ⚠️ **Ollama n'a pas d'endpoint
  de reranking natif** (feature request [#3368](https://github.com/ollama/ollama/issues/3368),
  toujours ouverte). Les cross-encodeurs type `bge-reranker` ont une tête de classification
  qu'Ollama ne sait pas servir. On utilise donc un LLM instruct comme juge de pertinence :
  pour chaque paire (requête, document) il renvoie un score 0–10 (normalisé en 0–1).
  Technique reconnue (RankGPT *pointwise*). Les appels sont lancés **en parallèle**
  (`asyncio.gather`) pour limiter la latence.

## Démarrage

Prérequis : Docker + Docker Compose.

```bash
docker compose up --build
```

Aucune configuration préalable requise : les valeurs par défaut sont dans le
`docker-compose.yml`. Le fichier `.env` est **optionnel** (uniquement pour surcharger
un paramètre, cf. `.env.example`).

Au premier lancement, `ollama-init` télécharge `bge-m3` (~1.2 Go) et `qwen3.5:4b` (~3.3 Go),
puis `indexer` calcule les embeddings et remplit Elasticsearch (s'exécute une fois et s'arrête).

- **Frontend** : http://localhost:3000
- **API (docs Swagger)** : http://localhost:8000/docs
- **Elasticsearch** : http://localhost:9200

### Exemple d'appel API

```bash
curl "http://localhost:8000/search?q=rénovation+toiture+école&top_k=5"
```

### Variante GPU (rapide)

Pour utiliser l'**Ollama natif de l'hôte** (GPU) au lieu du conteneur, mettre dans `.env` :
```
OLLAMA_BASE_URL=http://host.docker.internal:11434
```
(Les modèles doivent alors être présents côté hôte : `ollama pull bge-m3 && ollama pull qwen3.5:4b`.)
Pour le GPU *dans* le conteneur Ollama : décommenter le bloc `deploy` de `docker-compose.yml`.

## Développement (hors Docker)

```bash
# Elasticsearch + Ollama doivent tourner (localhost:9200 / :11434)
cd indexer  && pip install -r requirements.txt && python index.py
cd backend  && pip install -r requirements.txt && uvicorn main:app --reload
cd frontend && npm install && npm run dev      # http://localhost:5173
```

## Paramètres de recherche (`/search`)

| Param | Défaut | Description |
|---|---|---|
| `q` | — | Requête (obligatoire) |
| `probe` | 1 | Nombre de clusters sondés par kNN |
| `candidates` | 20 | Candidats BM25 récupérés avant rerank |
| `top_k` | 10 | Nombre de résultats renvoyés |
