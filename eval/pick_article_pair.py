import os
import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import re

# ----------------------
# Resolve project paths
# ----------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# ----------------------
# Load .env from root
# ----------------------
load_dotenv(PROJECT_ROOT / ".env")

# ----------------------
# Output directory
# ----------------------
OUTPUT_DIR = CURRENT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------
# DB config
# ----------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "wiki"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# ----------------------
# Argument parser
# ----------------------
parser = argparse.ArgumentParser(description="Pick an example article pair based on BERT score")
parser.add_argument(
    "--range",
    type=str,
    choices=["low", "medium", "high"],
    required=True,
    help="Select BERT score range (low/medium/high)"
)
args = parser.parse_args()

# ----------------------
# Connect to DB
# ----------------------
print("Connecting to database...")
conn = psycopg2.connect(**DB_CONFIG)

# ----------------------
# Load data
# ----------------------
query = """
SELECT 
    article_id,
    model_name,
    method,
    prompt_version,
    top_k,
    topic,
    generated_title,
    generated_text,
    reference_title,
    reference_text,
    bertscore_f1
FROM generated_articles
WHERE bertscore_f1 IS NOT NULL;
"""

df = pd.read_sql(query, conn)
print(f"Loaded {len(df)} rows from DB")

# ----------------------
# Select range
# ----------------------
bert_q1 = df["bertscore_f1"].quantile(0.33)
bert_q2 = df["bertscore_f1"].quantile(0.66)

if args.range == "low":
    subset = df[df["bertscore_f1"] <= bert_q1]
elif args.range == "medium":
    subset = df[(df["bertscore_f1"] > bert_q1) & (df["bertscore_f1"] <= bert_q2)]
else:  # high
    subset = df[df["bertscore_f1"] > bert_q2]

print(f"Selected {len(subset)} rows for range '{args.range}'")

# ----------------------
# Pick one random example
# ----------------------
example = subset.sample(n=1).iloc[0]

# ----------------------
# Cleaning function preserving paragraph breaks
# ----------------------
def clean_text(text: str) -> str:
    """
    Cleans text but keeps paragraphs:
    - Splits by double newlines
    - Removes extra whitespace and awkward line breaks within paragraphs
    - Joins paragraphs with double newlines
    """
    if not text:
        return ""
    paragraphs = re.split(r'\n\s*\n', text)
    cleaned_paragraphs = []
    for p in paragraphs:
        p_clean = re.sub(r'\s+', ' ', p).strip()
        if p_clean:
            cleaned_paragraphs.append(p_clean)
    return "\n\n".join(cleaned_paragraphs)

# ----------------------
# Prepare paths
# ----------------------
art_id = example["article_id"]
pair_dir = OUTPUT_DIR / f"art_pair_{art_id}"
pair_dir.mkdir(exist_ok=True)

orig_path = pair_dir / "original.txt"
gen_path = pair_dir / "generated.txt"

# ----------------------
# Save cleaned text
# ----------------------
original_clean = clean_text(example["reference_text"])
generated_clean = clean_text(example["generated_text"])

# Save original
with open(orig_path, "w", encoding="utf-8") as f:
    f.write(f"Title: {example['reference_title']}\n\n")
    f.write(original_clean)

# Save generated
with open(gen_path, "w", encoding="utf-8") as f:
    f.write(f"Title: {example['generated_title']}\n\n")
    f.write(generated_clean)

# ----------------------
# Done
# ----------------------
print(f"Example saved in {pair_dir}")
print(f"BERTScore: {example['bertscore_f1']:.4f}")
print(f"Topic: {example['topic']}, Model: {example['model_name']}, Method: {example['method']}")