import { useState, type FormEvent } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface Result {
  index: string;
  line: string;
  search_score: number;
  rerank_score: number;
}
interface SearchResponse {
  query: string;
  indices: string[];
  candidates: number;
  results: Result[];
}

export default function App() {
  const [query, setQuery] = useState("");
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/search?q=${encodeURIComponent(q)}`);
      if (!res.ok) throw new Error(`Erreur ${res.status}`);
      setData((await res.json()) as SearchResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header>
        <h1>🔎 IA Search — Consultations</h1>
        <p className="subtitle">
          clusters (knn) → recherche hybride → rerank LLM (logprobs)
        </p>
      </header>

      <form className="searchbar" onSubmit={runSearch}>
        <input
          type="text"
          value={query}
          placeholder="Ex. : rénovation de toiture d'école, formation sécurité…"
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "…" : "Search"}
        </button>
      </form>

      {error && <p className="error">⚠ {error}</p>}

      {data && (
        <section className="results">
          <p className="meta">
            {data.results.length} résultat(s) · index&nbsp;: <b>{data.indices.join(", ")}</b> ·{" "}
            {data.candidates} candidats
          </p>
          {data.results.map((r, i) => (
            <article key={i} className="card">
              <div className="line">{r.line}</div>
              <div className="badges">
                <span className="badge index">{r.index}</span>
                <span className="badge rerank" title="Score de pertinence (rerank logprobs)">
                  rerank {(r.rerank_score * 100).toFixed(0)}%
                </span>
                <span className="badge search" title="Score recherche hybride (knn + BM25)">
                  hybride {r.search_score}
                </span>
              </div>
              <div className="bar">
                <div className="fill" style={{ width: `${r.rerank_score * 100}%` }} />
              </div>
            </article>
          ))}
          {data.results.length === 0 && <p>Aucun résultat.</p>}
        </section>
      )}
    </div>
  );
}
