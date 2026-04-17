import csv
import random
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from main import load_profiles_from_csv, extract_skills, quality_score

# Expanded query templates with skill levels
SKILLS = [
    "python", "java", "javascript", "react", "node", "aws", "azure", "docker",
    "kubernetes", "sql", "machine learning", "django", "flask", "fastapi",
    "c++", "c#", "golang", "typescript", "pytorch", "tensorflow", "sap",
    "servicenow", "cybersecurity", "regulatory", "fda", "procurement",
    "supply chain", "data architecture", "etl", "siem", "leadership"
]

ROLES = [
    "developer", "engineer", "architect", "manager", "specialist",
    "analyst", "scientist", "consultant", "administrator"
]

LEVELS = ["junior", "mid", "senior", "lead", "principal"]

def generate_queries(n: int = 100) -> list:
    queries = []
    for _ in range(n):
        skill = random.choice(SKILLS)
        role = random.choice(ROLES)
        level = random.choice(LEVELS)
        if random.random() > 0.5:
            queries.append(f"hire {level} {skill} {role}")
        else:
            queries.append(f"{skill} {role} {level}")
    return list(set(queries))  # deduplicate

def is_boilerplate(text: str) -> bool:
    signals = ["basic knowledge", "foundational skills", "no explicit evidence"]
    return sum(s in text.lower() for s in signals) >= 2

def generate_clicks():
    profiles = load_profiles_from_csv("../profiles.csv")
    queries = generate_queries(200)  # generate 200 diverse queries
    print(f"Generated {len(queries)} unique queries.")

    rows = []
    for query in queries:
        query_skills = extract_skills(query)
        best_idx = -1
        best_score = -1
        for i, p in enumerate(profiles):
            if is_boilerplate(p["text"]):
                continue
            skill_overlap = len(set(p["entities"]).intersection(query_skills))
            # Score combines skill match, experience, and quality
            score = skill_overlap * 3 + p["years"] / 3 + quality_score(p["text"]) * 2
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx != -1:
            rows.append({"query": query, "positive_profile_id": best_idx})

    with open("clicks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "positive_profile_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} click entries in clicks.csv")

if __name__ == "__main__":
    generate_clicks()