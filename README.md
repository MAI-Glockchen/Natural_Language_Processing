# Natural_Language_Processing
All collaborative efforts for "Natural Language Processing" (Semester 2)

> **Python version:** 3.10 is required for this project.

This project extracts citations from Wikipedia articles, collects the referenced sources, cleans the text, and stores the passages in a **PostgreSQL database**.  
It serves as a foundation for later projects such as **Retrieval-Augmented Generation (RAG)** or Wikipedia article reconstruction.

---

# Installation
1. Activate the virtual environment

```bash
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare PostgreSQL:

```bash
docker compose up -d
```
If you don’t have Docker Desktop, make sure Docker Engine is installed and accessible via your terminal/WSL.

# Running the Pipeline
```bash
python main.py
```


