# IA Search — Moteur de recherche de consultations

Projet d'évaluation « Architectures IA ». Recherche sémantique sur un corpus de **1000
intitulés de marchés publics**, en **recherche hiérarchique** : embedding → cluster →
recherche hybride → reranking. Suit les TD1 à TD3 du module.

## Architecture

```
            ┌──────────────── INDEXATION (index_data) ────────────────┐
 dataset ─▶ Ollama (bge-m3) ─▶ embeddings ─▶ K-Means ─▶ Elasticsearch
                                                         ├─ data_clusters  { embedding, data_index }
                                                         └─ data_1 … data_5 { embedding, line }

            ┌──────────────────── RECHERCHE ──────────────────────────┐
 React ─▶ FastAPI ─▶ 1. embed requête (bge-m3)
                     2. knn sur "data_clusters"   → data_index du cluster le plus proche
                     3. recherche HYBRIDE (knn + BM25) dans data_X
                     4. rerank LLM par logprobs (qwen3.5), en parallèle
                     5. tri par pertinence → React
```

L'embedding **route** la requête vers un cluster (knn sur les centroïdes). La recherche
fine est **hybride** (vectorielle + BM25) dans l'index du cluster. Le **rerank** réordonne
les candidats par pertinence sémantique.

## Composants

| Dossier / service | Rôle | Techno |
|---|---|---|
| `index_data` | Script d'indexation (one-shot) | Python (uv), scikit-learn |
| `fastapi-backend` | API de recherche + rerank | FastAPI (uv), debugpy |
| `react-front` | Interface de recherche | React **TypeScript** (Vite) + nginx |
| `elasticsearch` | Base vectorielle + BM25 | Elasticsearch 8.12 |
| `ollama` / `ollama-init` | Modèles embedding + rerank | Ollama |

### Modèles & méthode

- **Embedding** : `bge-m3` (1024 dims) → `dense_vector` cosinus dans Elasticsearch.
- **Rerank** : LLM `qwen3.5` via Ollama, **scoring par logprobs** (TD3) : on contraint la
  réponse à `yes`/`no`, on lit les logprobs du token de réponse, on calcule
  `p_yes / (p_yes + p_no)`, puis tri décroissant. Appels **en parallèle** (`asyncio.gather`).
  Ollama n'ayant pas d'endpoint de rerank natif, c'est la méthode attendue par le cours.

## Démarrage

Prérequis : Docker + Docker Compose. Aucune configuration préalable.

```bash
# Détection automatique du GPU (GPU si NVIDIA dispo, sinon CPU) :
./start.sh          # Linux / macOS / Git Bash
./start.ps1         # Windows PowerShell

# ou directement (CPU, portable partout) :
docker compose up --build
```

Au premier lancement : téléchargement des modèles (`bge-m3`, `qwen3.5:4b`) par `ollama-init`,
puis indexation automatique par `index_data` (s'exécute une fois et s'arrête).

- **Frontend** : http://localhost:3000
- **API (Swagger)** : http://localhost:8000/docs
- **Elasticsearch** : http://localhost:9200
- **Debug backend** (debugpy) : port 5678 (Remote Attach, cf. TD2-3)

### Exemple d'appel

```bash
curl "http://localhost:8000/search?q=rénovation+toiture+école&top_k=5"
```

### GPU

`start.sh` / `start.ps1` détectent `nvidia-smi` et activent `docker-compose.gpu.yml`
(accès GPU du conteneur Ollama) si un GPU NVIDIA est présent. Sinon, exécution sur CPU.
Pour pointer vers l'Ollama **natif** de l'hôte : `OLLAMA_BASE_URL=http://host.docker.internal:11434` dans `.env`.

## Développement (hors Docker)

```bash
cd index_data      && uv run python main.py        # Elasticsearch + Ollama doivent tourner
cd fastapi-backend && uv run uvicorn main:app --reload
cd react-front     && npm install && npm run dev    # http://localhost:5173
```

## Paramètres de recherche (`/search`)

| Param | Défaut | Description |
|---|---|---|
| `q` | — | Requête (obligatoire) |
| `probe` | 1 | Nombre de clusters sondés (knn sur data_clusters) |
| `candidates` | 20 | Candidats de la recherche hybride avant rerank |
| `top_k` | 10 | Nombre de résultats renvoyés |
