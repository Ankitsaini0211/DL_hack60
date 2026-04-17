// src/utils/xaiExplanation.jsx
import { useState } from "react";

const XAIExplanationModule = ({ doc, query, bm25Score, denseScore, graphBoost, finalScore, entities = [] }) => {
  const [showDetails, setShowDetails] = useState(true);

  const total = parseFloat(bm25Score) + parseFloat(denseScore) + parseFloat(graphBoost) || 1;
  const bm25Pct = Math.round((parseFloat(bm25Score) / total) * 100);
  const densePct = Math.round((parseFloat(denseScore) / total) * 100);
  const graphPct = Math.round((parseFloat(graphBoost) / total) * 100);

  const queryTerms = query.toLowerCase().split(/\W+/).filter(t => t.length > 2);
  const matchedTerms = queryTerms.filter(term => 
    (doc.title + " " + doc.text).toLowerCase().includes(term)
  );

  const strongMatches = entities.filter(skill => 
    query.toLowerCase().includes(skill.toLowerCase()) || matchedTerms.some(t => skill.toLowerCase().includes(t))
  );

  return (
    <div style={{
      marginTop: 18,
      padding: 22,
      borderRadius: 16,
      background: "#0f172a",
      border: "1px solid #6366f1",
      color: "#e2e8f0"
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 26 }}>🔬</span>
          <div>
            <div style={{ fontSize: 17, fontWeight: 600, color: "#a5b4fc" }}>XAI Explanation Engine</div>
            <div style={{ fontSize: 13, color: "#94a3b8" }}>Transparent Retrieval Reasoning</div>
          </div>
        </div>
        <button onClick={() => setShowDetails(!showDetails)} style={{ padding: "6px 16px", borderRadius: 999, background: "#1e2937", border: "none", color: "#cbd5e1", fontSize: 13 }}>
          {showDetails ? "Hide" : "Show"} Details
        </button>
      </div>

      {/* Confidence Meter */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: 14 }}>
          <span>Match Confidence</span>
          <span style={{ color: parseFloat(finalScore) > 1.2 ? "#4ade80" : "#fbbf24", fontWeight: 600 }}>
            {parseFloat(finalScore) > 1.5 ? "EXCELLENT" : parseFloat(finalScore) > 1.0 ? "STRONG" : "GOOD"}
          </span>
        </div>
        <div style={{ height: 12, background: "#1e2937", borderRadius: 999, overflow: "hidden" }}>
          <div style={{ 
            height: "100%", 
            width: `${Math.min(100, Math.round(parseFloat(finalScore) * 45))}%`,
            background: "linear-gradient(90deg, #6366f1, #a855f7)",
            transition: "width 0.8s ease"
          }} />
        </div>
      </div>

      {/* Score Breakdown */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontWeight: 600, marginBottom: 12 }}>Signal Contributions</div>
        {[
          { label: "BM25 Keyword Matching", value: bm25Score, pct: bm25Pct, color: "#2563eb" },
          { label: "Dense Semantic Similarity", value: denseScore, pct: densePct, color: "#7c3aed" },
          { label: "GraphRAG Skill Relations", value: graphBoost, pct: graphPct, color: "#059669" }
        ].map((item, i) => (
          <div key={i} style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13.5, marginBottom: 4 }}>
              <span>{item.label}</span>
              <span><strong>{item.value}</strong> ({item.pct}%)</span>
            </div>
            <div style={{ height: 7, background: "#1e2937", borderRadius: 4 }}>
              <div style={{ height: "100%", width: `${item.pct}%`, background: item.color }} />
            </div>
          </div>
        ))}
      </div>

      {/* Strong Matches */}
      {strongMatches.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 10 }}>Strong Skill Matches</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {strongMatches.slice(0, 6).map((skill, i) => (
              <div key={i} style={{
                padding: "7px 16px",
                background: "rgba(5, 150, 105, 0.2)",
                color: "#4ade80",
                borderRadius: 999,
                fontSize: 13.5,
                border: "1px solid rgba(5, 150, 105, 0.4)"
              }}>
                ✓ {skill}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Natural Language Summary */}
      <div style={{
        padding: 18,
        background: "rgba(99, 102, 241, 0.12)",
        borderRadius: 12,
        borderLeft: "5px solid #6366f1",
        fontSize: 14.5,
        lineHeight: 1.7
      }}>
        This candidate was ranked highly because of <strong>strong keyword overlap</strong> in core skills, 
        <strong>semantic similarity</strong> in experience and context, and meaningful connections in the skill knowledge graph.
        {strongMatches.length > 0 && ` Particularly strong alignment found in ${strongMatches.slice(0, 2).join(" and ")}.`}
      </div>
    </div>
  );
};

export default XAIExplanationModule;