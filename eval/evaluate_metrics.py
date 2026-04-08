import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

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
# Connect to DB
# ----------------------
print("Connecting to database...")
conn = psycopg2.connect(**DB_CONFIG)

# ----------------------
# Queries
# ----------------------
query = """
SELECT 
    model_name,
    method,
    prompt_version,
    top_k,
    topic,
    rouge1_f1,
    rouge2_f1,
    rougel_f1,
    bertscore_f1,
    title_similarity,
    article_length_ratio
FROM generated_articles
WHERE bertscore_f1 IS NOT NULL;
"""

count_query = """
SELECT COUNT(*) 
FROM generated_articles 
WHERE bertscore_f1 IS NOT NULL;
"""

# ----------------------
# Get total row count
# ----------------------
total_rows = pd.read_sql(count_query, conn).iloc[0, 0]
print(f"Total rows to load: {total_rows}")

# ----------------------
# Load data with progress
# ----------------------
chunk_size = 1000
chunks = []

print("Loading data from DB...")
with tqdm(total=total_rows, desc="Rows loaded") as pbar:
    for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
        chunks.append(chunk)
        pbar.update(len(chunk))

df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df)} rows into DataFrame")

# ----------------------
# Save CSV
# ----------------------
csv_path = OUTPUT_DIR / "metrics.csv"
df.to_csv(csv_path, index=False)
print(f"Saved CSV to {csv_path}")

# ----------------------
# Plot settings
# ----------------------
sns.set(style="whitegrid")

# ----------------------
# 1. Model comparison (BERTScore)
# ----------------------
plt.figure(figsize=(10, 6))
model_scores = df.groupby("model_name")["bertscore_f1"].mean().sort_values()
sns.barplot(x=model_scores.values, y=model_scores.index)
plt.title("Average BERTScore by Model")
plt.xlabel("BERTScore F1")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png")
plt.close()

# ----------------------
# 2. Method comparison (BERTScore)
# ----------------------
plt.figure(figsize=(10, 6))
method_scores = df.groupby("method")["bertscore_f1"].mean().sort_values()
sns.barplot(x=method_scores.values, y=method_scores.index)
plt.title("Average BERTScore by Method")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "method_comparison.png")
plt.close()

# ----------------------
# 3. Distribution plots for BERTScore + ROUGE + title_similarity + article_length_ratio
# ----------------------
metrics_to_plot = [
    "bertscore_f1",
    "rouge1_f1",
    "rouge2_f1",
    "rougel_f1",
    "title_similarity",
    "article_length_ratio"
]

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[metric], bins=30, kde=True)
    plt.title(f"{metric.upper()} Distribution")
    plt.xlabel(metric.upper())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{metric}_distribution.png")
    plt.close()

# ----------------------
# 4. Correlation heatmap
# ----------------------
plt.figure(figsize=(10, 8))
corr = df[metrics_to_plot].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Metric Correlations")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

# ----------------------
# 5. Top results
# ----------------------
top = df.sort_values("bertscore_f1", ascending=False).head(10)
top.to_csv(OUTPUT_DIR / "top_results.csv", index=False)

# ----------------------
# Done
# ----------------------
print("\nAll done!")
print(f"Outputs saved in: {OUTPUT_DIR}")