// src/data/knowledgeGraph.js
export const KG_GRAPH = {
  nodes: [
    { id: "Python", type: "language", color: "#2563eb" },
    { id: "SQL", type: "language", color: "#7c3aed" },
    { id: "AWS", type: "cloud", color: "#db2777" },
    { id: "Azure", type: "cloud", color: "#0891b2" },
    { id: "SAP", type: "tool", color: "#d97706" },
    { id: "Regulatory Affairs", type: "domain", color: "#059669" },
    { id: "FDA", type: "regulation", color: "#dc2626" },
    { id: "Cybersecurity", type: "domain", color: "#7c3aed" },
    { id: "Supply Chain Management", type: "domain", color: "#059669" },
    { id: "Procurement", type: "domain", color: "#2563eb" },
    { id: "ServiceNow", type: "tool", color: "#6366f1" },
    { id: "ETL", type: "concept", color: "#0ea5e9" },
    { id: "SIEM", type: "tool", color: "#8b5cf6" },
    { id: "Leadership", type: "soft_skill", color: "#14b8a6" },
  ],
  edges: [
    { from: "Python", to: "AWS", label: "USED_IN" },
    { from: "Python", to: "SQL", label: "COMBINES_WITH" },
    { from: "SQL", to: "ETL", label: "USED_IN" },
    { from: "AWS", to: "Azure", label: "ALTERNATIVE_TO" },
    { from: "Supply Chain Management", to: "SAP", label: "POWERS" },
    { from: "Procurement", to: "Supply Chain Management", label: "PART_OF" },
    { from: "Regulatory Affairs", to: "FDA", label: "COMPLIES_WITH" },
    { from: "Cybersecurity", to: "SIEM", label: "USES" },
    { from: "ServiceNow", to: "Azure", label: "INTEGRATES_WITH" },
  ]
};