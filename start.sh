#!/usr/bin/env bash
# Lancement avec détection automatique du GPU (Linux / macOS / Git Bash).
set -e
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "GPU NVIDIA détecté -> Ollama sur GPU"
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
else
    echo "Pas de GPU NVIDIA -> Ollama sur CPU"
    docker compose up --build
fi
