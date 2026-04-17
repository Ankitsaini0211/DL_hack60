import React, { useState } from 'react';

const API_BASE = 'http://localhost:8000';

export default function HybridCandidateRetrieval() {
  const [query, setQuery] = useState('');
  const [alpha, setAlpha] = useState(0.3);
  const [useReranker, setUseReranker] = useState(true);
  const [useMMR, setUseMMR] = useState(true);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [intent, setIntent] = useState(null);
  const [expanded, setExpanded] = useState({});
  const [explaining, setExplaining] = useState({});

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResults([]);
    setIntent(null);
    setExpanded({});
    try {
      const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: 12,
          alpha,
          use_reranker: useReranker,
          use_mmr: useMMR,
        }),
      });
      if (!response.ok) throw new Error(`Search failed: ${response.status}`);
      const data = await response.json();
      setResults(data.results || []);
      setIntent(data.intent);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchExplanation = async (candidate) => {
    const id = candidate.id;
    setExplaining(prev => ({ ...prev, [id]: true }));
    try {
      const response = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, candidate }),
      });
      if (!response.ok) throw new Error('Explanation failed');
      const data = await response.json();
      setResults(prev =>
        prev.map(r => (r.id === id ? { ...r, explanation: data } : r))
      );
    } catch (err) {
      console.error(err);
    } finally {
      setExplaining(prev => ({ ...prev, [id]: false }));
    }
  };

  const toggleExplanation = async (candidate) => {
    const id = candidate.id;
    if (expanded[id]) {
      setExpanded(prev => ({ ...prev, [id]: false }));
    } else {
      if (!candidate.explanation) {
        await fetchExplanation(candidate);
      }
      setExpanded(prev => ({ ...prev, [id]: true }));
    }
  };

  return (
    <div style={{ padding: 20, fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      <h2>🔍 Industry‑Grade Hybrid Candidate Retrieval</h2>
      <p>Backend: BGE‑base‑en‑v1.5 + FAISS · BM25 fusion · Intent‑aware · Cross‑Encoder · MMR</p>

      <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSearch()}
          placeholder="e.g. Senior Python developer with AWS"
          style={{ flex: 1, padding: 12, fontSize: 16, borderRadius: 8, border: '1px solid #ccc' }}
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          style={{ padding: '12px 24px', fontSize: 16, background: '#2563eb', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer' }}
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 20, marginBottom: 15 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 14 }}>BM25 weight</span>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={alpha}
            onChange={e => setAlpha(parseFloat(e.target.value))}
            style={{ width: 150 }}
          />
          <span style={{ fontSize: 14 }}>Dense weight</span>
          <span style={{ marginLeft: 5, background: '#f3f4f6', padding: '4px 12px', borderRadius: 20, fontSize: 13 }}>
            α = {alpha.toFixed(2)} (BM25: {(alpha*100).toFixed(0)}% / Dense: {((1-alpha)*100).toFixed(0)}%)
          </span>
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={useReranker}
            onChange={e => setUseReranker(e.target.checked)}
          />
          <span style={{ fontSize: 14 }}>Cross‑Encoder Rerank</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={useMMR}
            onChange={e => setUseMMR(e.target.checked)}
          />
          <span style={{ fontSize: 14 }}>MMR Diversity</span>
        </label>
      </div>

      {intent && (
        <div style={{ marginBottom: 16, padding: '8px 16px', background: '#e0e7ff', borderRadius: 20, display: 'inline-block' }}>
          🧠 Detected intent: <strong>{intent}</strong>
        </div>
      )}

      {error && <div style={{ color: 'red', marginBottom: 16 }}>Error: {error}</div>}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
        {['Senior Python developer AWS', 'Regulatory affairs FDA compliance', 'Supply chain manager SAP', 'Cybersecurity SOC manager', 'Data architect ETL Spark'].map(q => (
          <button key={q} onClick={() => setQuery(q)} style={{ padding: '6px 12px', background: '#f3f4f6', border: '1px solid #d1d5db', borderRadius: 20, cursor: 'pointer' }}>
            {q}
          </button>
        ))}
      </div>

      {results.length > 0 && (
        <div>
          {results.map((candidate) => (
            <div key={candidate.id} style={{ marginBottom: 20, padding: 16, border: '1px solid #e5e7eb', borderRadius: 12, background: '#fff' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <h3 style={{ margin: 0 }}>{candidate.title}</h3>
                  <p style={{ color: '#4b5563', margin: '8px 0' }}>
                    {candidate.text}
                  </p>
                  <div style={{ fontSize: 14, color: '#6b7280', display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                    <span>BM25: {candidate.bm25_score?.toFixed(3) || 'N/A'}</span>
                    <span>Dense: {candidate.dense_score?.toFixed(3) || 'N/A'}</span>
                    {candidate.rerank_score !== undefined && (
                      <span>Rerank: <strong style={{ color: '#059669' }}>{candidate.rerank_score.toFixed(3)}</strong></span>
                    )}
                    <span>Final: <strong style={{ color: '#2563eb' }}>{candidate.score.toFixed(3)}</strong></span>
                  </div>
                  {candidate.entities?.length > 0 && (
                    <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {candidate.entities.slice(0, 8).map((ent, i) => (
                        <span key={i} style={{ background: '#e0e7ff', color: '#3730a3', padding: '2px 10px', borderRadius: 20, fontSize: 12 }}>
                          {ent}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => toggleExplanation(candidate)}
                  disabled={explaining[candidate.id]}
                  style={{ padding: '8px 16px', background: '#f9fafb', border: '1px solid #d1d5db', borderRadius: 8, cursor: 'pointer', marginLeft: 16 }}
                >
                  {explaining[candidate.id] ? 'Loading...' : (expanded[candidate.id] ? 'Hide' : 'Explain')}
                </button>
              </div>

              {expanded[candidate.id] && candidate.explanation && (
                <div style={{ marginTop: 16, padding: 16, background: '#f9fafb', borderRadius: 8 }}>
                  <h4 style={{ marginTop: 0 }}>🧠 Why this candidate?</h4>
                  <p>{candidate.explanation.why_retrieved}</p>
                  {candidate.explanation.key_terms?.length > 0 && (
                    <p><strong>Key terms:</strong> {candidate.explanation.key_terms.join(' · ')}</p>
                  )}
                  {candidate.explanation.retrieval_signals && (
                    <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
                      {Object.entries(candidate.explanation.retrieval_signals).map(([k, v]) => (
                        <div key={k} style={{ background: 'white', padding: '6px 12px', borderRadius: 6, textAlign: 'center' }}>
                          <div style={{ fontSize: 12, color: '#6b7280' }}>{k}</div>
                          <div style={{ fontWeight: 600 }}>{v}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {!loading && results.length === 0 && query && !error && (
        <p>No results found. Try a different query.</p>
      )}
    </div>
  );
}