"""
JD Scorer
=========
Drop a JD (PDF or TXT) into `job_description/`, run this script,
and get a ranked CSV report in `scoring_reports/`.

Flow:
    1. Detect latest JD file in job_description/
    2. Extract + clean JD text
    3. Use Ollama (LLaMA3) to parse JD → required skills, preferred skills, min experience
    4. Load all candidates from candidates.db
    5. Score each candidate  (skills 70% + experience 30%)
    6. Write ranked CSV to scoring_reports/

Requirements:
    pip install pdfplumber python-docx requests

Usage:
    python jd_scorer.py                        # scores using latest JD in folder
    python jd_scorer.py --jd path/to/jd.pdf   # scores using a specific JD file
    python jd_scorer.py --top 20               # show top 20 in terminal
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import requests

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    open  # built-in, always available
    HAS_TXT = True
except:
    HAS_TXT = False


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

JD_FOLDER      = Path("job_description")      # drop your JD files here
DB_PATH        = Path("candidates.db")        # your existing candidates DB
REPORTS_FOLDER = Path("scoring_reports")      # CSV reports written here

OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "llama3"                     # change to your local model name

# Scoring weights — must sum to 1.0
WEIGHT_SKILLS     = 0.70   # 70%  skill alignment
WEIGHT_EXPERIENCE = 0.30   # 30%  experience alignment


# ══════════════════════════════════════════════════════════════════════════
# SKILL NORMALISATION
# Identical to candidate_pipeline.py — required for matching to work.
# ══════════════════════════════════════════════════════════════════════════

SKILL_ALIASES: dict[str, str] = {
    "py": "python", "python3": "python",
    "js": "javascript", "node": "node.js", "nodejs": "node.js",
    "ms excel": "microsoft excel", "excel": "microsoft excel",
    "ms word": "microsoft word", "word": "microsoft word",
    "ms powerpoint": "microsoft powerpoint", "ppt": "microsoft powerpoint",
    "mysql": "mysql", "postgres": "postgresql", "postgre": "postgresql",
    "sklearn": "scikit-learn", "sci-kit learn": "scikit-learn",
    "tf": "tensorflow",
    "aws": "amazon web services", "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "powerbi": "power bi", "power-bi": "power bi",
    "cpp": "c++",
    "react.js": "react", "reactjs": "react",
    "vue.js": "vue", "vuejs": "vue",
    "git hub": "github",
    "k8s": "kubernetes",
    "nlp": "natural language processing",
    "ml": "machine learning", "ai": "artificial intelligence",
    "dl": "deep learning",
    "cv": "computer vision",
}

def normalise_skill(s: str) -> str:
    key = s.lower().strip()
    return SKILL_ALIASES.get(key, key)

def normalise_skills(skills: list[str]) -> set[str]:
    return {normalise_skill(s) for s in skills if s.strip()}


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — DETECT JD FILE
# ══════════════════════════════════════════════════════════════════════════

def get_latest_jd(folder: Path) -> Path:
    """Pick the most recently modified JD file in the folder."""
    supported = [".pdf", ".txt", ".docx", ".doc"]
    files = [f for f in folder.iterdir()
             if f.is_file() and f.suffix.lower() in supported]
    if not files:
        print(f"❌  No JD files found in '{folder}/'")
        sys.exit(1)
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"📄  Using JD: {latest.name}")
    return latest


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — EXTRACT + CLEAN JD TEXT
# ══════════════════════════════════════════════════════════════════════════

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()

    if ext == ".pdf":
        if not HAS_PDF:
            print("❌  pdfplumber not installed. Run: pip install pdfplumber")
            sys.exit(1)
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)

    elif ext == ".txt":
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="replace")

    else:
        print(f"❌  Unsupported file type: {ext}. Only PDF and TXT are accepted.")
        sys.exit(1)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)       # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank lines
    text = re.sub(r"={3,}|-{3,}|\*{3,}", "", text)  # remove dividers
    text = re.sub(r"apply\s+(now|here)[^\n]*", "", text, flags=re.I)  # CTA noise
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — PARSE JD VIA OLLAMA
# ══════════════════════════════════════════════════════════════════════════

JD_PROMPT = """You are a recruitment assistant. Extract structured information from the job description below.

Return ONLY a valid JSON object with this exact structure:
{{
  "role": "job title as a string",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "min_experience": 0,
  "education": "Bachelor's or Master's or PhD or null"
}}

Rules:
- required_skills: skills explicitly marked as required, must-have, or mandatory
- preferred_skills: skills marked as preferred, nice-to-have, bonus, or desirable
- min_experience: minimum years as a plain number (e.g. 3 for "3+ years"), 0 if not mentioned
- education: only the minimum level required, null if not mentioned
- Return ONLY the JSON object. No explanation. No markdown. No extra text.

Job Description:
<<<
{jd_text}
>>>
"""

def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        print(f"❌  Cannot connect to Ollama at {OLLAMA_URL}")
        print("    Make sure Ollama is running: ollama serve")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌  Ollama request timed out after 120s")
        sys.exit(1)


def parse_jd_with_ollama(jd_text: str) -> dict:
    prompt  = JD_PROMPT.format(jd_text=jd_text[:4000])
    raw     = call_ollama(prompt)

    # Strip markdown fences in case model adds them
    cleaned = re.sub(r"```json|```", "", raw).strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        print("❌  Ollama did not return valid JSON. Raw response:")
        print(raw[:500])
        sys.exit(1)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"❌  JSON parse error: {e}\n    Raw: {raw[:500]}")
        sys.exit(1)

    jd_profile = {
        "role":             data.get("role", "Unknown Role"),
        "required_skills":  normalise_skills(data.get("required_skills", [])),
        "preferred_skills": normalise_skills(data.get("preferred_skills", [])),
        "min_experience":   float(data.get("min_experience") or 0),
        "education":        data.get("education") or None,
    }

    print(f"✅  JD parsed")
    print(f"    Role          : {jd_profile['role']}")
    print(f"    Required  ({len(jd_profile['required_skills'])}): {', '.join(sorted(jd_profile['required_skills']))}")
    print(f"    Preferred ({len(jd_profile['preferred_skills'])}): {', '.join(sorted(jd_profile['preferred_skills']))}")
    print(f"    Min experience: {jd_profile['min_experience']} yrs")
    return jd_profile


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — LOAD CANDIDATES FROM candidates.db
# ══════════════════════════════════════════════════════════════════════════

def load_candidates(db_path: Path) -> list[dict]:
    if not db_path.exists():
        print(f"❌  Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, filename, name, skills, experience_years, roles, education, summary
        FROM candidates
    """)
    rows = cursor.fetchall()
    conn.close()

    candidates = []
    for row in rows:
        # skills stored as JSON string: '["python", "sql", "aws"]'
        try:
            raw_skills = json.loads(row["skills"] or "[]")
        except (json.JSONDecodeError, TypeError):
            raw_skills = []
        try:
            roles = json.loads(row["roles"] or "[]")
            # flatten if Ollama stored roles as dicts e.g. [{"title": "Data Analyst"}]
            if roles and isinstance(roles[0], dict):
                roles = [r.get("title") or r.get("role") or str(r) for r in roles]
        except (json.JSONDecodeError, TypeError):
            roles = [row["roles"]] if row["roles"] else []

        try:
            education = json.loads(row["education"] or "null")
        except (json.JSONDecodeError, TypeError):
            education = row["education"] or None

        candidates.append({
            "id":               row["id"],
            "filename":         row["filename"] or "",
            "name":             row["name"] or "Unknown",
            "skills":           normalise_skills(raw_skills),  # normalised set
            "experience_years": float(row["experience_years"] or 0),
            "roles":            roles if isinstance(roles, list) else [roles],
            "education":        education,
            "summary":          row["summary"] or "",
        })

    print(f"👥  Loaded {len(candidates)} candidates")
    return candidates


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — SCORE EACH CANDIDATE
# ══════════════════════════════════════════════════════════════════════════

def score_candidate(candidate: dict, jd: dict) -> dict:
    """
    Score one candidate against the JD.

    Skills score (70%):
        - Base  = matched required skills / total required skills
        - Bonus = up to +15% for preferred skill matches
        - Capped at 100%

    Experience score (30%):
        - 100%  if candidate meets or exceeds min_experience
        - 75%   if within 25% below requirement
        - 50%   if within 50% below requirement
        - Pro-rated below that
    """
    cand_skills = candidate["skills"]       # normalised set
    req_skills  = jd["required_skills"]     # normalised set
    pref_skills = jd["preferred_skills"]    # normalised set
    min_exp     = jd["min_experience"]
    cand_exp    = candidate["experience_years"]

    # ── Skills ────────────────────────────────────────────────────────────
    if req_skills:
        matched_req = cand_skills & req_skills
        req_ratio   = len(matched_req) / len(req_skills)
    else:
        matched_req = set()
        req_ratio   = 1.0

    matched_pref = cand_skills & pref_skills
    pref_bonus   = (len(matched_pref) / len(pref_skills) * 0.15) if pref_skills else 0.0
    skill_score  = min(1.0, req_ratio + pref_bonus)

    # ── Experience ────────────────────────────────────────────────────────
    if min_exp <= 0:
        exp_score = 1.0
    elif cand_exp >= min_exp:
        exp_score = 1.0
    elif cand_exp >= min_exp * 0.75:
        exp_score = 0.75
    elif cand_exp >= min_exp * 0.5:
        exp_score = 0.50
    else:
        exp_score = max(0.0, cand_exp / min_exp)

    # ── Weighted total ────────────────────────────────────────────────────
    total = (skill_score * WEIGHT_SKILLS) + (exp_score * WEIGHT_EXPERIENCE)

    # ── Breakdown strings ─────────────────────────────────────────────────
    missing_req = req_skills - cand_skills
    exp_gap     = max(0.0, min_exp - cand_exp)

    edu = candidate["education"]
    if isinstance(edu, list):
        edu = edu[0] if edu else "—"
    edu = str(edu) if edu else "—"

    return {
        "rank":               0,           # filled after sorting
        "name":               candidate["name"],
        "total_score":        round(total * 100, 1),
        "skills_score":       round(skill_score * 100, 1),
        "experience_score":   round(exp_score * 100, 1),
        "required_match_pct": f"{round(req_ratio * 100)}%",
        "matched_required":   ", ".join(sorted(matched_req))  or "—",
        "missing_required":   ", ".join(sorted(missing_req))  or "—",
        "matched_preferred":  ", ".join(sorted(matched_pref)) or "—",
        "candidate_exp_yrs":  cand_exp,
        "jd_min_exp_yrs":     min_exp,
        "experience_gap":     f"-{exp_gap:.1f} yrs" if exp_gap > 0 else "meets requirement",
        "education":          edu,
        "roles":              ", ".join(candidate["roles"]) if candidate["roles"] else "—",
        "summary":            (candidate["summary"] or "")[:200],
        "filename":           candidate["filename"],
        "candidate_id":       candidate["id"],
    }


def score_all(candidates: list[dict], jd: dict) -> list[dict]:
    results = [score_candidate(c, jd) for c in candidates]
    results.sort(key=lambda r: r["total_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    return results


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — WRITE CSV REPORT
# ══════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "rank",
    "name",
    "total_score",
    "candidate_exp_yrs",
    "summary",
    "filename",
    "candidate_id",
]

def write_csv(results: list[dict], jd: dict, jd_filename: str) -> Path:
    REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    role_slug  = re.sub(r"[^\w]+", "_", jd["role"].lower()).strip("_")
    out_path   = REPORTS_FOLDER / f"scored_{role_slug}_{timestamp}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # Metadata block at top of file (lines starting with # are ignored by Excel)
        f.write(f"# JD Scoring Report\n")
        f.write(f"# JD File        : {jd_filename}\n")
        f.write(f"# Role           : {jd['role']}\n")
        f.write(f"# Min Experience : {jd['min_experience']} years\n")
        f.write(f"# Required Skills: {', '.join(sorted(jd['required_skills']))}\n")
        f.write(f"# Preferred Skills: {', '.join(sorted(jd['preferred_skills']))}\n")
        f.write(f"# Weights        : Skills {int(WEIGHT_SKILLS*100)}%  |  Experience {int(WEIGHT_EXPERIENCE*100)}%\n")
        f.write(f"# Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Candidates     : {len(results)}\n")
        f.write("#\n")

        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📊  Report saved → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], top_n: int = 10):
    n = min(top_n, len(results))
    print(f"\n{'─'*72}")
    print(f"  TOP {n} CANDIDATES  (out of {len(results)})")
    print(f"{'─'*72}")
    print(f"  {'#':<4} {'Name':<26} {'Total':>6} {'Skills':>7} {'Exp':>6}  Req Match")
    print(f"{'─'*72}")
    for r in results[:n]:
        print(
            f"  {r['rank']:<4} {r['name']:<26} "
            f"{r['total_score']:>5.1f}  "
            f"{r['skills_score']:>6.1f}  "
            f"{r['experience_score']:>5.1f}  "
            f"{r['required_match_pct']}"
        )
    print(f"{'─'*72}\n")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Score candidates against a Job Description")
    parser.add_argument(
        "--jd",
        type=str,
        default=None,
        help="Path to a specific JD file. Defaults to the latest file in job_description/",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DB_PATH),
        help=f"Path to candidates SQLite DB (default: {DB_PATH})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to print in terminal (default: 10)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)

    # Step 1 — find JD
    if args.jd:
        jd_path = Path(args.jd)
        if not jd_path.exists():
            print(f"❌  JD file not found: {jd_path}")
            sys.exit(1)
        print(f"📄  Using JD: {jd_path.name}")
    else:
        JD_FOLDER.mkdir(parents=True, exist_ok=True)
        jd_path = get_latest_jd(JD_FOLDER)

    # Step 2 — extract + clean
    print("\n⏳  Extracting JD text …")
    raw  = extract_text(jd_path)
    text = clean_text(raw)
    print(f"    {len(text)} characters extracted")

    # Step 3 — parse via Ollama
    print(f"\n🤖  Parsing JD with Ollama ({MODEL}) …")
    jd_profile = parse_jd_with_ollama(text)

    # Step 4 — load candidates
    print(f"\n📂  Loading candidates from {db_path.name} …")
    candidates = load_candidates(db_path)
    if not candidates:
        print("⚠️   No candidates in the database. Run candidate_pipeline.py first.")
        sys.exit(0)

    # Step 5 — score
    print(f"\n⚙️   Scoring …")
    results = score_all(candidates, jd_profile)

    # Step 6 — output
    write_csv(results, jd_profile, jd_path.name)
    print_summary(results, top_n=args.top)


if __name__ == "__main__":
    main()