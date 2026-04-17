# cleaned_profiles.py
import csv
import json

def clean_profiles(csv_path="profiles.csv"):
    profiles = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('id') or str(row['id']).strip() == '':
                continue

            name = (row.get('name') or f"Candidate_{row['id']}").strip()
            if name.lower() in ['name test', 'technical skills', '']:
                name = f"Candidate {row['id']}"

            potential_roles = (row.get('potential_roles') or '').split(',')[0].strip()
            title = f"{name} – {potential_roles or 'Professional Profile'}"

            # Rich text for retrieval
            skill_summary = (row.get('skill_summary') or '').strip()
            if not skill_summary:
                skills = []
                for col in ['core_skills', 'secondary_skills', 'soft_skills']:
                    if row.get(col):
                        skills.extend([s.strip() for s in row[col].split(',') if s.strip()])
                skill_summary = "Skills: " + ", ".join(list(dict.fromkeys(skills)))

            # Clean entities
            entities = set()
            for col in ['core_skills', 'secondary_skills', 'soft_skills']:
                if row.get(col):
                    for skill in row[col].split(','):
                        clean = skill.strip().replace('(', '').replace(')', '').replace('"', '').strip()
                        if clean and len(clean) > 2:
                            entities.add(clean)

            profiles.append({
                "id": str(row['id']).strip(),
                "title": title,
                "text": skill_summary,
                "entities": list(entities)[:18]
            })

    print(f"✅ Cleaned {len(profiles)} profiles successfully!")
    
    # Save as JS file
    with open("src/data/cleanedProfiles.js", "w", encoding="utf-8") as f:
        f.write("// Auto-generated from profiles.csv\n")
        f.write("export const cleanedProfiles = ")
        f.write(json.dumps(profiles, indent=2))
        f.write(";\n")
    
    return profiles

if __name__ == "__main__":
    clean_profiles()