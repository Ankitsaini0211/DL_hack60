import { useState, useCallback } from "react";
import { cleanedProfiles } from "../data/cleanedProfiles";
import { KG_GRAPH } from "../data/knowledgeGraph";
import XAIExplanationModule from "../utils/xaiExplanation";

export default function HybridCandidateRetrievalSystem() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [activeStep, setActiveStep] = useState(-1);
  const [isSearching, setIsSearching] = useState(false);
  const [intent, setIntent] = useState(null);
  const [expandedQueries, setExpandedQueries] = useState([]);
  const [activeTab, setActiveTab] = useState("search");
  const [metrics, setMetrics] = useState(null);
  const [graphNodes, setGraphNodes] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [retrievalMode, setRetrievalMode] = useState("hybrid");

  const avgLen = cleanedProfiles.reduce((s, d) => 
    s + (d.title + " " + d.text).split(/\W+/).length, 0
  ) / cleanedProfiles.length;

  // ====================== HELPER FUNCTIONS ======================
  function betterSemanticEmbed(text) {
    const words = text.toLowerCase().split(/\W+/).filter(Boolean);
    const importantTerms = ["python", "sql", "aws", "azure", "sap", "regulatory", "fda", "procurement", "supply chain", "cybersecurity", "servicenow", "etl", "siem", "leadership"];
    
    let vector = new Array(importantTerms.length + 5).fill(0);
    
    words.forEach((word, idx) => {
      const termIdx = importantTerms.indexOf(word);
      if (termIdx !== -1) vector[termIdx] += 1.5;
      else if (word.length > 4) vector[importantTerms.length] += 0.8;
      if (idx < 15) vector[importantTerms.length + 1] += 0.3;
    });
    
    const norm = Math.sqrt(vector.reduce((a, b) => a + b * b, 0)) || 1;
    return vector.map(v => v / norm);
  }

  function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
  }

  function bm25Score(query, doc, avgLen) {
    const k1 = 1.5, b = 0.75;
    const qTerms = query.toLowerCase().split(/\W+/).filter(Boolean);
    const dTerms = (doc.title + " " + doc.text).toLowerCase().split(/\W+/);
    const dLen = dTerms.length;
    let score = 0;

    qTerms.forEach(q => {
      const df = cleanedProfiles.filter(d => (d.title + " " + d.text).toLowerCase().includes(q)).length;
      if (!df) return;
      const idf = Math.log((cleanedProfiles.length - df + 0.5) / (df + 0.5) + 1);
      const tf = dTerms.filter(t => t === q).length;
      score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dLen / avgLen));
    });
    return score;
  }

  function rrfFuse(lists, k = 60) {
    const scores = {};
    lists.forEach(list => {
      list.forEach((id, rank) => {
        scores[id] = (scores[id] || 0) + 1 / (k + rank + 1);
      });
    });
    return Object.entries(scores).sort((a, b) => b[1] - a[1]);
  }

  function detectIntent(query) {
    const q = query.toLowerCase();
    if (/\b(python|sql|java|aws|azure|servicenow|siem|etl)\b/.test(q))
      return { type: "navigational", label: "Skill/Tech Match", mode: "bm25", color: "#2563eb", bm25Weight: 0.65, denseWeight: 0.25 };
    if (/\b(why|how|experience|background|profile)\b/.test(q))
      return { type: "informational", label: "Candidate Profile", mode: "dense", color: "#7c3aed", bm25Weight: 0.3, denseWeight: 0.6 };
    if (/\b(compare|versus|vs|best|strongest)\b/.test(q))
      return { type: "analytical", label: "Multi-candidate Comparison", mode: "graph", color: "#059669", bm25Weight: 0.4, denseWeight: 0.4 };
    return { type: "hybrid", label: "General Candidate Search", mode: "hybrid", color: "#d97706", bm25Weight: 0.5, denseWeight: 0.4 };
  }

  function expandQuery(query) {
    let expanded = [query];
    const q = query.toLowerCase();
    const rules = {
      "developer": ["software engineer", "full stack", "backend engineer"],
      "python": ["django", "flask", "data science", "pandas"],
      "aws": ["cloud infrastructure", "ec2", "s3"],
      "supply": ["procurement", "logistics", "inventory management"],
      "regulatory": ["compliance", "fda", "cosmetics regulation"],
      "cyber": ["soc", "siem", "soar", "firewall"],
    };
    Object.keys(rules).forEach(key => {
      if (q.includes(key)) expanded.push(...rules[key]);
    });
    return [...new Set(expanded)];
  }

  // ====================== MAIN SEARCH ======================
  const runSearch = useCallback(async () => {
    if (!query.trim()) return;

    setIsSearching(true);
    setResults([]);
    setMetrics(null);
    setActiveStep(0);

    await new Promise(r => setTimeout(r, 300));
    const det = detectIntent(query);
    setIntent(det);
    setActiveStep(1);

    await new Promise(r => setTimeout(r, 300));
    const exqs = expandQuery(query);
    setExpandedQueries(exqs);
    setActiveStep(2);

    const bm25Scores = cleanedProfiles.map(doc => ({
      doc, score: bm25Score(query, doc, avgLen)
    })).sort((a, b) => b.score - a.score);

    const bm25Ids = bm25Scores.map(x => x.doc.id);

    await new Promise(r => setTimeout(r, 400));
    const qEmb = betterSemanticEmbed(query);
    const denseScores = cleanedProfiles.map(doc => ({
      doc, score: cosineSim(qEmb, betterSemanticEmbed(doc.title + " " + doc.text))
    })).sort((a, b) => b.score - a.score);

    const denseIds = denseScores.map(x => x.doc.id);

    await new Promise(r => setTimeout(r, 400));
    const fused = rrfFuse([bm25Ids.slice(0, 15), denseIds.slice(0, 15)]);

    const finalScores = fused.map(([id]) => {
      const doc = cleanedProfiles.find(d => d.id === id);
      if (!doc) return null;

      const bm25Val = bm25Scores.find(s => s.doc.id === id)?.score || 0;
      const denseVal = denseScores.find(s => s.doc.id === id)?.score || 0;

      let graphBoost = doc.entities?.some(e => graphNodes.includes(e)) ? 0.15 : 0;

      const finalScore = (det.bm25Weight * bm25Val) + (det.denseWeight * denseVal) + graphBoost;

      return {
        ...doc,
        bm25Score: bm25Val.toFixed(3),
        denseScore: denseVal.toFixed(3),
        graphBoost: graphBoost.toFixed(2),
        finalScore: finalScore.toFixed(3)
      };
    }).filter(Boolean);

    const reranked = finalScores.sort((a, b) => parseFloat(b.finalScore) - parseFloat(a.finalScore)).slice(0, 8);

    setResults(reranked);
    setActiveStep(6);

    const related = [...new Set(reranked.flatMap(d => d.entities || []))].slice(0, 10);
    setGraphNodes(related);
    setActiveStep(7);

    setIsSearching(false);
    setActiveStep(-1);
  }, [query, avgLen, graphNodes]);

  return (
    <div style={{ fontFamily: "var(--font-sans)", maxWidth: 1000, margin: "0 auto", padding: "2rem 1rem", color: "var(--color-text-primary)" }}>
      <h1 style={{ fontSize: 26, fontWeight: 600, marginBottom: 8 }}>Hybrid Candidate Retrieval</h1>
      <p style={{ color: "#94a3b8", marginBottom: 24 }}>
        Using {cleanedProfiles.length} real profiles • BM25 + Dense + GraphRAG + XAI
      </p>

      {/* Search Bar */}
      <div style={{ display: "flex", gap: 12, marginBottom: 30 }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === "Enter" && runSearch()}
          placeholder="e.g. Python developer with AWS experience"
          style={{
            flex: 1, padding: "14px 18px", fontSize: 16, borderRadius: 12,
            border: "2px solid #475569", background: "#0f172a", color: "#e2e8f0"
          }}
        />
        <button onClick={runSearch} disabled={isSearching || !query.trim()}
          style={{ padding: "14px 32px", borderRadius: 12, background: "#6366f1", color: "white", border: "none", fontWeight: 600 }}>
          {isSearching ? "Searching..." : "Search"}
        </button>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div>
          {results.map(doc => (
            <div key={doc.id} style={{
              marginBottom: 24, padding: 20, borderRadius: 16,
              border: "1px solid #334155", background: "#0f172a"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <div>
                  <h3 style={{ margin: "0 0 8px 0", fontSize: 17 }}>{doc.title}</h3>
                  <p style={{ color: "#cbd5e1", fontSize: 14.5, lineHeight: 1.6 }}>
                    {doc.text.slice(0, 210)}...
                  </p>
                  <div style={{ marginTop: 12, display: "flex", gap: 20, fontSize: 14 }}>
                    <span>BM25: <strong>{doc.bm25Score}</strong></span>
                    <span>Dense: <strong>{doc.denseScore}</strong></span>
                    <span>Graph: <strong>+{doc.graphBoost}</strong></span>
                    <span style={{ color: "#a5b4fc" }}>Final: {doc.finalScore}</span>
                  </div>
                </div>
                <button 
                  onClick={() => setSelectedResult(selectedResult?.id === doc.id ? null : doc)}
                  style={{ padding: "10px 20px", borderRadius: 10, background: "#1e2937", border: "1px solid #6366f1", color: "#e0e7ff" }}
                >
                  {selectedResult?.id === doc.id ? "Hide" : "Explain"} ↗
                </button>
              </div>

              {selectedResult?.id === doc.id && (
                <XAIExplanationModule
                  doc={doc}
                  query={query}
                  bm25Score={doc.bm25Score}
                  denseScore={doc.denseScore}
                  graphBoost={doc.graphBoost}
                  finalScore={doc.finalScore}
                  entities={doc.entities || []}
                />
              )}
            </div>
          ))}
        </div>
      )}

      {results.length === 0 && !isSearching && (
        <div style={{ textAlign: "center", padding: "100px 20px", color: "#64748b" }}>
          Enter a query above to start hybrid search on your full candidate dataset
        </div>
      )}
    </div>
  );
}