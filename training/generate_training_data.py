import csv
import random
import re
from typing import List, Dict

# ------------------------------------------------------------
# Helper functions (must match main.py logic)
# ------------------------------------------------------------
def is_boilerplate(text: str) -> bool:
    """Return True if text contains multiple low-quality signals."""
    signals = [
        "basic knowledge",
        "foundational skills",
        "entry-level",
        "no explicit evidence",
        "familiar with",
    ]
    count = sum(1 for s in signals if s in text.lower())
    return count >= 2

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9+#]+", text.lower())

def quality_score(text: str) -> float:
    """Statistical quality score (matches main.py)."""
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    has_numbers = 1.0 if re.search(r'\d+', text) else 0.0
    return 0.6 * min(unique_ratio / 0.7, 1.0) + 0.4 * has_numbers

# ------------------------------------------------------------
# Load profiles from CSV
# ------------------------------------------------------------
def load_profiles(csv_path: str) -> List[Dict]:
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
                            entities.add(clean.lower())
            profiles.append({
                "id": row["id"],
                "text": row["skill_summary"],
                "entities": list(entities),
                "years": float(row.get("years_of_experience", 0) or 0),
            })
    return profiles

# ------------------------------------------------------------
# Synthetic queries and expected intent/skills
# ------------------------------------------------------------
QUERY_TEMPLATES = [
    ("hire senior python developer", "hire", ["python"]),
    ("java backend engineer", "hire", ["java"]),
    ("aws cloud architect", "hire", ["aws"]),
    ("learn python programming", "learn", ["python"]),
    ("cybersecurity expert", "hire", ["cybersecurity"]),
    ("data engineer with etl", "hire", ["etl"]),
    ("supply chain manager", "hire", ["supply chain"]),
    ("regulatory affairs fda", "hire", ["regulatory", "fda"]),
    ("entry level software developer", "hire", []),
    ("compare python vs java", "compare", ["python", "java"]),
]

def compute_features(profile: Dict, query: str, query_skills: List[str]) -> Dict:
    # BM25 (simplified)
    doc_tokens = tokenize(profile["text"])
    query_tokens = tokenize(query)
    bm25 = sum(1 for t in query_tokens if t in doc_tokens) / (len(query_tokens) + 1)

    # Dense (simulated as skill overlap + random)
    skill_overlap = len(set(profile["entities"]).intersection(set(query_skills)))
    dense = 0.5 + 0.3 * skill_overlap + random.uniform(-0.1, 0.1)

    # Quality
    quality = quality_score(profile["text"])

    # Intent confidence (simulated)
    intent_conf = 0.8 if any(s in profile["text"].lower() for s in query_skills) else 0.5

    return {
        "bm25": round(bm25, 3),
        "dense": round(min(max(dense, 0), 1), 3),
        "years": profile["years"],
        "skill_overlap": skill_overlap,
        "quality": round(quality, 3),
        "intent_conf": round(intent_conf, 3),
    }

def generate_relevance(profile: Dict, query: str, query_skills: List[str], intent: str) -> int:
    # Boilerplate gets zero relevance
    if is_boilerplate(profile["text"]):
        return 0

    skill_match = len(set(profile["entities"]).intersection(set(query_skills)))
    years = profile["years"]
    text_lower = profile["text"].lower()

    if intent == "hire":
        if years >= 5 and skill_match >= 2:
            return 5
        elif years >= 2 and skill_match >= 1:
            return 3
        elif skill_match >= 1:
            return 1
        else:
            return 0
    elif intent == "learn":
        if "beginner" in text_lower or "entry" in text_lower:
            return 3 if skill_match >= 1 else 1
        else:
            return 1
    else:
        return 2 if skill_match >= 1 else 1

def main():
    profiles = load_profiles("../profiles.csv")
    print(f"Loaded {len(profiles)} profiles.")

    with open("training_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "bm25", "dense", "years", "skill_overlap", "quality", "intent_conf", "relevance"])

        query_id = 0
        for q_text, intent, skills in QUERY_TEMPLATES:
            query_id += 1
            # Sample 30 random profiles per query
            sampled = random.sample(profiles, min(30, len(profiles)))
            for profile in sampled:
                feats = compute_features(profile, q_text, skills)
                rel = generate_relevance(profile, q_text, skills, intent)
                writer.writerow([
                    query_id,
                    feats["bm25"],
                    feats["dense"],
                    feats["years"],
                    feats["skill_overlap"],
                    feats["quality"],
                    feats["intent_conf"],
                    rel
                ])
        print(f"Generated training data for {query_id} queries.")

if __name__ == "__main__":
    main()