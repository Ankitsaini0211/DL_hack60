# =========================
# Hybrid Retrieval System (Improved)
# =========================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# =========================
# STEP 1: Load Data
# =========================

df = pd.read_csv("profiles.csv")

print("Columns:", df.columns)
print("Total profiles:", len(df))


# =========================
# STEP 2: Prepare Text
# =========================

def create_text(row):
    return f"""
    name: {row.get('name', '')}
    core skills: {row.get('core_skills', '')}
    secondary skills: {row.get('secondary_skills', '')}
    soft skills: {row.get('soft_skills', '')}
    experience: {row.get('years_of_experience', '')} years
    roles: {row.get('potential_roles', '')}
    summary: {row.get('skill_summary', '')}
    """

def clean_text(text):
    return text.lower().replace(",", " ").replace("\n", " ")

texts = df.apply(create_text, axis=1).tolist()
texts = [clean_text(t) for t in texts]

print("\nSample text:\n", texts[0][:300])


# =========================
# STEP 3: Embedding Model (UPGRADED)
# =========================

print("\nLoading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en")

print("Generating embeddings...")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS index ready")


# =========================
# STEP 4: BM25
# =========================

tokenized_corpus = [text.split() for text in texts]
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 ready")


# =========================
# STEP 5: Improved Scoring
# =========================

def hybrid_score(semantic_rank, bm25_rank, alpha=0.7):
    s_score = 1 / (1 + semantic_rank) if semantic_rank is not None else 0
    b_score = 1 / (1 + bm25_rank) if bm25_rank is not None else 0
    return alpha * s_score + (1 - alpha) * b_score


# =========================
# STEP 6: Domain Boosting
# =========================

IMPORTANT_WORDS = ["healthcare", "finance", "nlp", "vision", "ai"]

def boost_score(text, query):
    boost = 0
    for word in IMPORTANT_WORDS:
        if word in query and word in text:
            boost += 0.05
    return boost


# =========================
# STEP 7: Explanation
# =========================

def get_explanation(query, row):
    explanation = []

    query = query.lower()

    if "machine learning" in query and "machine learning" in str(row["core_skills"]).lower():
        explanation.append("ML match")

    if "nlp" in query and "nlp" in str(row["core_skills"]).lower():
        explanation.append("NLP match")

    if "healthcare" in query and "healthcare" in str(row["skill_summary"]).lower():
        explanation.append("Healthcare domain")

    if row["years_of_experience"] >= 3:
        explanation.append("Experienced")

    return explanation


# =========================
# STEP 8: Hybrid Search
# =========================

def hybrid_search(query, top_k=5):
    query_clean = clean_text(query)

    # ----- Semantic -----
    query_vec = model.encode([query_clean])
    D, I = index.search(np.array(query_vec), top_k)

    semantic_ranks = {idx: rank for rank, idx in enumerate(I[0])}

    # ----- BM25 -----
    tokenized_query = query_clean.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

    bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_indices)}

    # ----- Combine -----
    all_indices = set(semantic_ranks) | set(bm25_ranks)

    final_scores = {}

    for idx in all_indices:
        score = hybrid_score(
            semantic_ranks.get(idx),
            bm25_ranks.get(idx)
        )

        # 🔥 Domain boost
        score += boost_score(texts[idx], query_clean)

        # 🔥 Experience boost
        exp = df.iloc[idx]["years_of_experience"]
        if not pd.isna(exp):
            score += 0.01 * float(exp)

        final_scores[idx] = score

    # Sort
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    results = []

    for idx, score in sorted_results[:top_k]:
        results.append({
            "name": df.iloc[idx].get("name", ""),
            "skills": df.iloc[idx].get("core_skills", ""),
            "experience": df.iloc[idx].get("years_of_experience", 0),
            "roles": df.iloc[idx].get("potential_roles", ""),
            "score": score,
            "explanation": get_explanation(query_clean, df.iloc[idx])
        })

    return results


# =========================
# STEP 9: Run
# =========================

if __name__ == "__main__":

    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            break

        results = hybrid_search(query)

        print("\nTop Results:\n")

        for i, res in enumerate(results):
            print(f"Rank {i+1} | Score: {res['score']:.4f}")
            print(f"Name: {res['name']}")
            print(f"Skills: {res['skills']}")
            print(f"Experience: {res['experience']} years")
            print(f"Roles: {res['roles']}")
            print(f"Why selected: {res['explanation']}")
            print("-" * 60)