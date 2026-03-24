import os
import re
import json
import sqlite3
import requests
from uuid import uuid4
from datetime import datetime

# Optional parsers
try:
    import pdfplumber
    HAS_PDF = True
except:
    HAS_PDF = False

try:
    from docx import Document
    HAS_DOCX = True
except:
    HAS_DOCX = False



INPUT_FOLDER = "candidate_profile"
DB_PATH = "candidates.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3" 



conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    filename TEXT,
    name TEXT,
    skills TEXT,
    experience_years REAL,
    roles TEXT,
    education TEXT,
    summary TEXT,
    processed_at TEXT
)
""")
conn.commit()


# CHECK DUPLICATES 
def is_processed(filename):
    cursor.execute("SELECT 1 FROM candidates WHERE filename=?", (filename,))
    return cursor.fetchone() is not None


# EXT EXTRACTION 
def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf" and HAS_PDF:
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)

    elif ext in [".docx", ".doc"] and HAS_DOCX:
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)

    else:
        with open(filepath, "r", errors="ignore") as f:
            return f.read()


#CLEAN TEXT
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# PROMPT
def build_prompt(resume_text):
    return f"""

Resume:
<<<
{resume_text[:4000]}
>>>
"""


#OLLAMA
def call_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    return response.json().get("response", "")


# ---------------- PARSE JSON ----------------
def parse_json(text):
    text = re.sub(r"```json|```", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    return json.loads(text)


# ---------------- VALIDATE OUTPUT ----------------
def validate_output(data):
    return {
        "name": data.get("name", "Unknown"),
        "skills": list(set([s.lower() for s in data.get("skills", [])])),
        "experience_years": float(data.get("experience_years", 0)),
        "roles": data.get("roles", []),
        "education": data.get("education", []),
        "summary": data.get("summary", "")[:500]
    }


# ---------------- SAVE TO DB ----------------
def save_to_db(filename, data):
    cursor.execute("""
    INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid4()),
        filename,
        data["name"],
        json.dumps(data["skills"]),
        data["experience_years"],
        json.dumps(data["roles"]),
        json.dumps(data["education"]),
        data["summary"],
        datetime.utcnow().isoformat()
    ))
    conn.commit()


# ---------------- MAIN PIPELINE ----------------
def process_candidates():
    for file in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, file)

        if not os.path.isfile(path):
            continue

        if is_processed(file):
            print(f"Skipped: {file}")
            continue

        print(f"Processing: {file}")

        try:
            text = extract_text(path)

            if not text.strip():
                print(f"Empty file: {file}")
                continue

            cleaned = clean_text(text)
            prompt = build_prompt(cleaned)

            # Retry mechanism
            for attempt in range(2):
                try:
                    raw = call_ollama(prompt)
                    parsed = parse_json(raw)
                    validated = validate_output(parsed)
                    save_to_db(file, validated)
                    print(f"Saved: {file}")
                    break
                except Exception as e:
                    print(f"Retry {attempt+1} failed: {e}")
                    if attempt == 1:
                        print(f"Failed completely: {file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    process_candidates()