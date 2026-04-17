# generate_real_embeddings.py   ← Replace your old file with this
import json
from sentence_transformers import SentenceTransformer

print("Loading model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load cleanedProfiles.js and extract only the array
with open("src/data/cleanedProfiles.js", "r", encoding="utf-8") as f:
    content = f.read()

# Find the JSON array part
start = content.find('[')
end = content.rfind(']') + 1
if start == -1 or end == 0:
    raise ValueError("Could not find profiles array in cleanedProfiles.js")

profiles = json.loads(content[start:end])

print(f"Found {len(profiles)} profiles. Generating real embeddings...")

# Create richer text for better semantic quality
documents = []
for p in profiles:
    rich_text = f"Role: {p.get('title', '')}\nSkills: {', '.join(p.get('entities', [])[:20])}\nSummary: {p.get('text', '')}"
    documents.append(rich_text)

# Generate embeddings
embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True)

# Save as clean JSON
data = {
    "profiles": profiles,
    "embeddings": embeddings.tolist()
}

with open("src/data/realEmbeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)   # indent=2 makes it readable

print("✅ Successfully created clean realEmbeddings.json")
print(f"   Profiles: {len(profiles)}")
print(f"   Embedding dimension: {embeddings.shape[1]}")