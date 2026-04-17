import re
from typing import Tuple, List

# Intent keywords
INTENT_KEYWORDS = {
    "hire": ["hire", "recruit", "looking for", "need", "senior", "lead", "manager", "developer", "engineer"],
    "learn": ["learn", "course", "tutorial", "how to", "guide", "beginner"],
    "compare": ["compare", "vs", "versus", "difference", "better"],
    "debug": ["error", "bug", "fix", "issue", "crash"],
}

# Known skills (extend as needed)
KNOWN_SKILLS = {
    "python", "java", "javascript", "react", "node", "aws", "azure", "docker",
    "kubernetes", "sql", "machine learning", "django", "flask", "fastapi",
    "c++", "c#", "ruby", "golang", "typescript", "pytorch", "tensorflow",
    "sap", "servicenow", "cybersecurity", "regulatory", "fda", "procurement",
    "supply chain", "data architecture", "etl", "siem", "leadership"
}

def classify_intent(query: str) -> Tuple[str, float]:
    """Classify query intent and return confidence."""
    query_lower = query.lower()
    scores = {intent: 0.0 for intent in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', query_lower):
                scores[intent] += 1.0
    best_intent = max(scores, key=scores.get)
    max_score = scores[best_intent]
    if max_score == 0.0:
        return "explore", 0.5
    confidence = min(0.95, 0.5 + (max_score * 0.1))
    return best_intent, confidence

def extract_skills(query: str) -> List[str]:
    """Extract known skill entities from query."""
    query_lower = query.lower()
    extracted = []
    for skill in KNOWN_SKILLS:
        if re.search(rf'\b{re.escape(skill)}\b', query_lower):
            extracted.append(skill)
    return extracted