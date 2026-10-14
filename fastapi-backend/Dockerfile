FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
EXPOSE 8000 5678
# debugpy écoute sur 5678 (debug à distance, cf. TD2-3) sans bloquer le démarrage.
CMD ["uv", "run", "-m", "debugpy", "--listen", "0.0.0.0:5678", \
     "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
