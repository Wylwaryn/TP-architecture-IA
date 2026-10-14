import { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runSearch(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      if (!res.ok) throw new Error(`Erreur ${res.status}`);
      setData(await res.json());
    } catch (err) {
      setError(err.message);
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
          Recherche hiérarchique&nbsp;: embedding&nbsp;→ cluster&nbsp;→ BM25&nbsp;→ rerank LLM
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
          {loading ? "…" : "Rechercher"}
        </button>
      </form>

      {error && <p className="error">⚠ {error}</p>}

      {data && (
        <section className="results">
          <p className="meta">
            {data.results.length} résultat(s) · cluster(s) sondé(s)&nbsp;:{" "}
            <b>{data.clusters.join(", ")}</b> · {data.candidates} candidats BM25
          </p>
          {data.results.map((r) => (
            <article key={r.line_id} className="card">
              <div className="line">{r.line}</div>
              <div className="badges">
                <span className="badge cluster">cluster {r.cluster_id}</span>
                <span className="badge rerank" title="Score de pertinence du reranker LLM">
                  rerank {(r.rerank_score * 100).toFixed(0)}%
                </span>
                <span className="badge bm25" title="Score lexical BM25">
                  BM25 {r.bm25_score}
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
