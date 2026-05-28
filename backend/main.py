import json
import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import lightgbm as lgb
import tantivy
import requests  # for LLM calls (optional)

# ---------- Configuration ----------
FINE_TUNED_MODEL_PATH = "./fine_tuned_retriever"
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
DATA_PATH = "../profiles.csv"
LTR_MODEL_PATH = "lambdarank_model.txt"
TANTIVY_INDEX_PATH = "./tantivy_index"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")
OLLAMA_URL = "http://localhost:11434/api/generate"

# Choose model: fine-tuned if available, else base
if os.path.exists(FINE_TUNED_MODEL_PATH):
    MODEL_NAME = FINE_TUNED_MODEL_PATH
    print("✅ Using fine-tuned retriever model.")
else:
    MODEL_NAME = BASE_MODEL_NAME
    print("ℹ️ Fine-tuned model not found, using base BGE.")

app = FastAPI(title="Production Hybrid Candidate Retrieval")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Global resources ----------
model = None
reranker = None
profiles = []
embeddings = None
faiss_index = None
tantivy_index = None
ranker = None
neo4j_driver = None

# ---------- Query Understanding (embedding-based) ----------
_intent_prototypes = {
    "hire": "hire recruit looking for senior lead manager engineer developer",
    "learn": "learn course tutorial how to guide beginner",
    "compare": "compare vs versus difference better",
    "explore": "explore discover find browse search",
}
_prototype_embs = {}

def classify_intent_fast(query: str, model: SentenceTransformer) -> Tuple[str, float]:
    global _prototype_embs
    if not _prototype_embs:
        for intent, text in _intent_prototypes.items():
            _prototype_embs[intent] = model.encode([text], normalize_embeddings=True)[0]
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    best_intent = "explore"
    best_score = -1.0
    for intent, proto_emb in _prototype_embs.items():
        score = float(np.dot(q_emb, proto_emb))
        if score > best_score:
            best_score = score
            best_intent = intent
    confidence = min(0.95, 0.5 + best_score * 0.3)
    return best_intent, confidence

def extract_skills(text: str) -> List[str]:
    skills_keywords = {
        "python", "java", "javascript", "react", "node", "aws", "azure", "docker",
        "kubernetes", "sql", "machine learning", "django", "flask", "fastapi",
        "c++", "c#", "ruby", "golang", "typescript", "pytorch", "tensorflow",
        "sap", "servicenow", "cybersecurity", "regulatory", "fda", "procurement",
        "supply chain", "data architecture", "etl", "siem", "leadership"
    }
    text_lower = text.lower()
    return [s for s in skills_keywords if s in text_lower]

# ---------- Query Expansion ----------
def expand_query(query: str, intent: str, skills: List[str]) -> List[str]:
    variations = [query]
    if intent == "hire":
        variations.append(f"senior {query}")
        variations.append(f"lead {query}")
        variations.append(f"experienced {query}")
    if skills:
        variations.append(" ".join(skills[:3]) + " expert")
    # Add LLM-based expansion if Ollama is available (optional)
    try:
        prompt = f"Rewrite the search query '{query}' for recruiting into 2 alternative variations. Output one per line."
        resp = requests.post(OLLAMA_URL, json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=5)
        if resp.status_code == 200:
            lines = resp.json()["response"].strip().split("\n")
            variations.extend([line.strip() for line in lines if line.strip()])
    except:
        pass  # fallback to rule-based only
    return list(set(variations))

# ---------- Profile Loading ----------
def load_profiles_from_csv(csv_path: str) -> List[Dict]:
    import csv
    profiles = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id") or not row.get("skill_summary"):
                continue
            entities = set()
            for col in ["core_skills", "secondary_skills", "soft_skills"]:
                if row.get(col):
                    for skill in row[col].split(","):
                        clean = re.sub(r'\([^)]*\)', '', skill).strip()
                        if clean:
                            entities.add(clean)
            profiles.append({
                "id": row["id"],
                "name": row.get("name", f"Candidate {row['id']}"),
                "title": f"{row.get('name', 'Candidate')} – {row.get('potential_roles', '').split(',')[0] if row.get('potential_roles') else 'Professional'}",
                "text": row["skill_summary"],
                "entities": list(entities),
                "years": float(row.get("years_of_experience", 0) or 0),
            })
    return profiles

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9+#]+", text.lower())

# ---------- Index Building ----------
def build_tantivy_index(profiles: List[Dict]) -> tantivy.Index:
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("id", stored=True)
    schema_builder.add_text_field("text", stored=True, tokenizer_name="en_stem")
    schema = schema_builder.build()
    os.makedirs(TANTIVY_INDEX_PATH, exist_ok=True)
    if os.path.exists(os.path.join(TANTIVY_INDEX_PATH, "meta.json")):
        print("✅ Opening existing Tantivy index...")
        return tantivy.Index.open(TANTIVY_INDEX_PATH)
    print("🛠️ Building new Tantivy index...")
    index = tantivy.Index(schema, path=TANTIVY_INDEX_PATH)
    writer = index.writer()
    for i, p in enumerate(profiles):
        writer.add_document(tantivy.Document(id=str(i), text=p["text"]))
    writer.commit()
    return index

def build_indices():
    global model, reranker, profiles, embeddings, faiss_index, tantivy_index, ranker, neo4j_driver
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Loading reranker model...")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("Loading profiles...")
    profiles = load_profiles_from_csv(DATA_PATH)
    print(f"Loaded {len(profiles)} profiles.")
    docs = [p["text"] for p in profiles]
    emb_path = "profiles.npy"
    faiss_path = "faiss_index.bin"
    embeddings = None
    if os.path.exists(emb_path) and os.path.exists(faiss_path):
        try:
            temp_embeddings = np.load(emb_path)
            if temp_embeddings.shape[0] == len(profiles):
                print("✅ Loading existing pre-computed embeddings and FAISS index...")
                embeddings = temp_embeddings
                faiss_index = faiss.read_index(faiss_path)
            else:
                raise ValueError(f"Cache count ({temp_embeddings.shape[0]}) mismatch with loaded profiles ({len(profiles)})")
        except Exception as e:
            print(f"ℹ️ Re-generating embeddings due to cache mismatch or load error: {e}")
            embeddings = None

    if embeddings is None:
        print("Generating embeddings from scratch...")
        embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
        np.save(emb_path, embeddings)
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings.astype(np.float32))
        faiss.write_index(faiss_index, faiss_path)
        print("💾 Saved embeddings and FAISS index cache.")
    print("Building Tantivy BM25 index...")
    tantivy_index = build_tantivy_index(profiles)
    if os.path.exists(LTR_MODEL_PATH):
        ranker = lgb.Booster(model_file=LTR_MODEL_PATH)
        print("LightGBM ranker loaded.")
    else:
        ranker = None
        print("LTR model not found; using heuristic fallback.")
    # Try to connect to Neo4j
    try:
        from neo4j import GraphDatabase
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        neo4j_driver.verify_connectivity()
        print("✅ Connected to Neo4j.")
    except Exception as e:
        neo4j_driver = None
        print(f"⚠️ Neo4j not available: {e}")
    print("Indices ready.")

build_indices()

# ---------- Retrieval Helpers ----------
def bm25_search(query: str, top_k: int = 100) -> List[tuple]:
    searcher = tantivy_index.searcher()
    q, _ = tantivy_index.parse_query_lenient(query, ["text"])
    hits = searcher.search(q, top_k).hits
    return [(int(searcher.doc(doc_address)["id"][0]), score) for score, doc_address in hits]

def dense_search(query: str, top_k: int = 100) -> List[tuple]:
    emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = faiss_index.search(emb, top_k)
    return [(int(i), float(d)) for i, d in zip(indices[0], distances[0])]

def graph_search(skills: List[str], top_k: int = 50) -> List[tuple]:
    if not neo4j_driver or not skills:
        return []
    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
            WHERE s.name IN $skills
            RETURN c.id AS id, COUNT(s) AS matches
            ORDER BY matches DESC
            LIMIT $top_k
            """,
            skills=skills, top_k=top_k
        )
        return [(record["id"], record["matches"]) for record in result]

def is_boilerplate(text: str) -> bool:
    signals = ["basic knowledge", "foundational skills", "entry-level", "no explicit evidence", "familiar with", "due to limited information"]
    count = sum(1 for s in signals if s in text.lower())
    if len(text.split()) < 40:
        count += 1
    return count >= 2

def quality_score(text: str) -> float:
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    has_numbers = 1.0 if re.search(r'\d+', text) else 0.0
    return 0.6 * min(unique_ratio / 0.7, 1.0) + 0.4 * has_numbers

def cross_encoder_rerank(query: str, candidates: List[dict], top_k: int = 30) -> List[dict]:
    if not candidates:
        return []
    pairs = [(query, cand["text"]) for cand in candidates]
    ce_scores = reranker.predict(pairs)
    for cand, ce_score in zip(candidates, ce_scores):
        qual = quality_score(cand["text"])
        blend_weight = 0.8 if qual > 0.6 else 0.3
        cand["score"] = blend_weight * float(ce_score) + (1 - blend_weight) * cand["score"]
        cand["rerank_score"] = float(ce_score)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

def mmr_rerank(candidates: List[dict], lambda_param: float = 0.7) -> List[dict]:
    if len(candidates) <= 1:
        return candidates
    cand_embs = model.encode([c["text"] for c in candidates], normalize_embeddings=True)
    selected = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < len(candidates):
        mmr_scores = []
        for idx in remaining:
            sim_query = candidates[idx]["score"]
            max_sim = 0.0
            for sel_idx in selected:
                sim = float(np.dot(cand_embs[idx], cand_embs[sel_idx]))
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_param * sim_query - (1 - lambda_param) * max_sim
            mmr_scores.append((idx, mmr))
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]

def normalize_scores(candidates: List[dict]) -> List[dict]:
    if not candidates:
        return candidates
    scores = [c["score"] for c in candidates]
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s > 1e-6:
        for c in candidates:
            c["score"] = (c["score"] - min_s) / (max_s - min_s)
    return candidates

# ---------- LLM Answer Generation ----------
def generate_answer(query: str, candidates: List[dict]) -> str:
    if not candidates:
        return "No suitable candidates found."
    context = "\n\n".join([f"Candidate {i+1}: {c['title']}\n{c['text']}" for i, c in enumerate(candidates[:3])])
    prompt = f"""You are a recruiting assistant. Based on the following candidate profiles, answer the query: "{query}". 
Provide a concise summary and recommend the best match. Include key skills and experience.

{context}
"""
    try:
        resp = requests.post(OLLAMA_URL, json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=15)
        if resp.status_code == 200:
            return resp.json()["response"].strip()
    except:
        pass
    return "LLM generation unavailable. Please review the candidate list manually."

# ---------- API Models ----------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 12
    use_reranker: bool = True
    use_mmr: bool = True
    generate_answer: bool = False

class ExplainRequest(BaseModel):
    query: str
    candidate: Dict[str, Any]

# ---------- Endpoints ----------
@app.post("/search")
async def search(request: SearchRequest):
    try:
        # 1. Query Understanding
        intent, intent_conf = classify_intent_fast(request.query, model)
        query_skills = extract_skills(request.query)

        # 2. Multi-query expansion
        expanded_queries = expand_query(request.query, intent, query_skills)
        print(f"Expanded queries: {expanded_queries}")

        # 3. Recall (BM25 + Dense + Graph) over all expanded queries
        all_bm25, all_dense, all_graph = [], [], []
        for q in expanded_queries:
            all_bm25.extend(bm25_search(q, top_k=50))
            all_dense.extend(dense_search(q, top_k=50))
        graph_candidates = graph_search(query_skills, top_k=50)
        all_graph = [(profiles.index(next(p for p in profiles if p["id"] == cid)), score) for cid, score in graph_candidates]

        # Merge and deduplicate
        bm25_dict, dense_dict, graph_dict = {}, {}, {}
        for idx, score in all_bm25:
            bm25_dict[idx] = max(bm25_dict.get(idx, 0), score)
        for idx, score in all_dense:
            dense_dict[idx] = max(dense_dict.get(idx, 0), score)
        for idx, score in all_graph:
            graph_dict[idx] = max(graph_dict.get(idx, 0), score)

        # Hard boilerplate filter
        combined_indices = set(bm25_dict.keys()) | set(dense_dict.keys()) | set(graph_dict.keys())
        combined_indices = [idx for idx in combined_indices if not is_boilerplate(profiles[idx]["text"])]

        # 4. First-stage Ranking (LTR or heuristic)
        candidates = []
        if ranker:
            features_list, idx_list = [], []
            for idx in combined_indices:
                p = profiles[idx]
                features_list.append([
                    bm25_dict.get(idx, 0.0),
                    dense_dict.get(idx, 0.0),
                    p.get("years", 0),
                    len(set(p.get("entities", [])).intersection(query_skills)),
                    quality_score(p["text"]),
                    intent_conf
                ])
                idx_list.append(idx)
            ltr_scores = ranker.predict(np.array(features_list))
            for i, score in enumerate(ltr_scores):
                idx = idx_list[i]
                candidates.append({
                    "index": idx,
                    "score": float(score),
                    "bm25_score": bm25_dict.get(idx, 0.0),
                    "dense_score": dense_dict.get(idx, 0.0),
                    "graph_score": graph_dict.get(idx, 0.0),
                })
        else:
            # Heuristic fallback with min-max normalization
            bm25_vals = list(bm25_dict.values())
            dense_vals = list(dense_dict.values())
            graph_vals = list(graph_dict.values())
            min_b, max_b = min(bm25_vals) if bm25_vals else (0, 1)
            min_d, max_d = min(dense_vals) if dense_vals else (0, 1)
            min_g, max_g = min(graph_vals) if graph_vals else (0, 1)
            for idx in combined_indices:
                bm25_norm = (bm25_dict.get(idx, 0) - min_b) / (max_b - min_b + 1e-6)
                dense_norm = (dense_dict.get(idx, 0) - min_d) / (max_d - min_d + 1e-6)
                graph_norm = (graph_dict.get(idx, 0) - min_g) / (max_g - min_g + 1e-6)
                qual = quality_score(profiles[idx]["text"])
                years = profiles[idx].get("years", 0)
                score = 0.3 * bm25_norm + 0.5 * dense_norm + 0.1 * graph_norm + 0.1 * qual + 0.05 * min(years/10.0, 1.0)
                candidates.append({
                    "index": idx,
                    "score": score,
                    "bm25_score": bm25_dict.get(idx, 0),
                    "dense_score": dense_dict.get(idx, 0),
                    "graph_score": graph_dict.get(idx, 0),
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:50]

        # Build result objects
        results = []
        for c in top_candidates:
            p = profiles[c["index"]]
            results.append({
                "id": p["id"],
                "title": p["title"],
                "text": p["text"][:400] + ("..." if len(p["text"]) > 400 else ""),
                "entities": p.get("entities", [])[:10],
                "score": c["score"],
                "bm25_score": c.get("bm25_score", 0),
                "dense_score": c.get("dense_score", 0),
                "graph_score": c.get("graph_score", 0),
                "years": p.get("years", 0),
                "full_text": p["text"],
            })

        # 5. Neural Reranker
        if request.use_reranker and results:
            results = cross_encoder_rerank(request.query, results, top_k=30)

        # 6. MMR Diversity
        if request.use_mmr and results:
            results = mmr_rerank(results, lambda_param=0.7)

        # 7. Normalize final scores and trim
        final_results = normalize_scores(results[:request.top_k])
        for r in final_results:
            r.pop("full_text", None)

        # 8. LLM Answer Generation (optional)
        answer = None
        if request.generate_answer:
            answer = generate_answer(request.query, final_results)

        return {
            "results": final_results,
            "intent": intent,
            "expanded_queries": expanded_queries,
            "answer": answer,
            "query": request.query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain(request: ExplainRequest):
    cand = request.candidate
    q_skills = extract_skills(request.query)
    matched = [s for s in cand.get("entities", []) if s.lower() in [qs.lower() for qs in q_skills]]
    return {
        "why_retrieved": f"This candidate matches {len(matched)} skill(s) from your query: {', '.join(matched[:5])}.",
        "key_terms": matched[:5],
        "confidence": "high" if len(matched) >= 2 else "medium",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)