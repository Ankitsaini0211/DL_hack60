import os
import random
import pandas as pd
import numpy as np
import re
from typing import List

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
import faiss
import tantivy
import csv

# ---------- CONFIG ----------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
DATA_PATH = "../profiles.csv"
TANTIVY_INDEX_PATH = "../backend/tantivy_index"
OUTPUT_PATH = "../backend/fine_tuned_retriever"
CLICKS_PATH = "clicks.csv"

# ---------- UTILS (mirror main.py) ----------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9+#]+", text.lower())

def quality_score(text: str) -> float:
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    has_numbers = 1.0 if re.search(r'\d+', text) else 0.0
    return 0.6 * min(unique_ratio / 0.7, 1.0) + 0.4 * has_numbers

def is_boilerplate(text: str) -> bool:
    signals = [
        "basic knowledge",
        "foundational skills",
        "entry-level",
        "no explicit evidence",
        "familiar with",
    ]
    count = sum(1 for s in signals if s in text.lower())
    if len(text.split()) < 40:
        count += 1
    return count >= 2

def extract_skills_from_text(text: str) -> List[str]:
    skills_keywords = {
        "python", "java", "javascript", "react", "node", "aws", "azure", "docker",
        "kubernetes", "sql", "machine learning", "django", "flask", "fastapi",
        "c++", "c#", "ruby", "golang", "typescript", "pytorch", "tensorflow",
        "sap", "servicenow", "cybersecurity", "regulatory", "fda", "procurement",
        "supply chain", "data architecture", "etl", "siem", "leadership"
    }
    text_lower = text.lower()
    return [s for s in skills_keywords if s in text_lower]

def load_profiles(csv_path: str) -> List[dict]:
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
                "text": row["skill_summary"],
                "entities": list(entities),
                "years": float(row.get("years_of_experience", 0) or 0),
            })
    return profiles

# ---------- LOAD DATA ----------
print("📄 Loading profiles...")
profiles = load_profiles(DATA_PATH)
print(f"Loaded {len(profiles)} profiles.")

# ---------- LOAD MODEL ----------
print("🧠 Loading base model...")
model = SentenceTransformer(MODEL_NAME)

# ---------- BUILD FAISS (dense index) ----------
print("⚡ Encoding profiles for FAISS...")
docs = [p["text"] for p in profiles]
embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings.astype(np.float32))

# ---------- LOAD TANTIVY (BM25 index) ----------
print("🔎 Loading Tantivy index...")
tantivy_index = tantivy.Index.open(TANTIVY_INDEX_PATH)

def bm25_search(query: str, top_k: int = 50) -> List[int]:
    searcher = tantivy_index.searcher()
    parser = tantivy.QueryParser.for_index(tantivy_index, ["text"])
    q = parser.parse_query(query)
    hits = searcher.search(q, top_k).hits
    return [int(searcher.doc(doc_id)["id"][0]) for doc_id, _ in hits]

def dense_search(query: str, top_k: int = 50) -> List[int]:
    emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = faiss_index.search(emb, top_k)
    return list(indices[0])

# ---------- HARD NEGATIVE MINING ----------
def mine_hard_negatives(query: str, pos_idx: int, top_k: int = 50) -> List[int]:
    bm25_ids = bm25_search(query, top_k)
    dense_ids = dense_search(query, top_k)
    pool = list(set(bm25_ids + dense_ids))

    negatives = []
    for idx in pool:
        if idx == pos_idx:
            continue
        if is_boilerplate(profiles[idx]["text"]):
            continue
        if quality_score(profiles[idx]["text"]) > 0.3:
            negatives.append(idx)

    if len(negatives) == 0:
        return []
    return random.sample(negatives, min(5, len(negatives)))

# ---------- BUILD TRAINING DATASET ----------
def build_training_dataset() -> List[InputExample]:
    if not os.path.exists(CLICKS_PATH):
        raise FileNotFoundError(f"{CLICKS_PATH} not found. Run generate_clicks.py first.")

    df = pd.read_csv(CLICKS_PATH)
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        pos_idx = int(row["positive_profile_id"])
        pos_text = profiles[pos_idx]["text"]
        neg_ids = mine_hard_negatives(query, pos_idx)
        for neg_idx in neg_ids:
            neg_text = profiles[neg_idx]["text"]
            examples.append(InputExample(texts=[query, pos_text, neg_text]))
    print(f"✅ Built {len(examples)} triplets.")
    return examples

# ---------- TRAIN ----------
def train():
    examples = build_training_dataset()
    if not examples:
        print("No training examples. Exiting.")
        return

    train_model = SentenceTransformer(MODEL_NAME)
    loader = DataLoader(examples, shuffle=True, batch_size=32)
    loss = MultipleNegativesRankingLoss(train_model)

    print("🏋️ Training for 3 epochs...")
    train_model.fit(
        train_objectives=[(loader, loss)],
        epochs=3,
        warmup_steps=100,
        show_progress_bar=True,
        output_path=OUTPUT_PATH
    )
    print(f"✅ Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    train()