# Wikipedia Citation Pipeline

> **Python version:** 3.14 is required for this project.

[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://www.docker.com/)

## 📋 Overview

This project extracts citations from Wikipedia articles, collects the referenced sources, cleans the text, and stores the passages in a **PostgreSQL database**.

It serves as a foundation for later projects such as **Retrieval-Augmented Generation (RAG)** or Wikipedia article reconstruction.

---

## 🚀 Quick Start (Docker - Recommended)

**No installation required!** Just use Docker:

```bash
# Start both PostgreSQL and the Python application
docker compose up -d

# View application logs
docker compose logs -f app

# Stop everything
docker compose down
```

The application will automatically:
- Build the Python container with all dependencies
- Connect to the PostgreSQL database
- Start processing Wikipedia articles

---

## ⚙️ Configuration

You can customize the application by setting environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_URL` | `postgresql+psycopg2://user:password@db:5432/wiki_db` | PostgreSQL connection string |
| `RATE_LIMIT_DELAY` | `1.0` | Seconds between requests |
| `MAX_WORKERS` | `4` | Number of parallel workers |
| `CACHE_DIR` | `./cache` | Directory for caching |
| `CACHE_TTL` | `3600` | Cache time-to-live in seconds |
| `MAX_SENTENCES_PER_PASSAGE` | `3` | Max sentences per passage |
| `MIN_TEXT_LENGTH` | `200` | Minimum text length for valid content |

### Example: Custom Configuration

```bash
# Run with custom settings
docker compose up -d --build \
  --env-file .env.custom
```

---

## 🛠️ Running the Pipeline

### Command Line Arguments

```bash
python main.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--target-count` | `420` | Number of articles to collect |
| `--min-citations` | `20` | Minimum citations per article |
| `--max-docs` | `50` | Maximum citations to process per article |
| `--workers` | `4` | Number of parallel workers |

### Examples

```bash
# Run with default settings
python main.py

# Run with custom parameters
python main.py --target-count 100 --min-citations 10 --workers 2

# Run with specific article URL
python main.py --target-count 1 --min-citations 50
```

---

## 📁 Project Structure

```
Natural_Language_Processing/
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Python application container
├── requirements.txt        # Python dependencies
├── main.py                 # Application entry point
├── config.py               # Configuration settings
├── README.md               # This file
├── db/                     # Database schema/migrations
├── pipeline/               # Core processing logic
│   ├── __init__.py
│   ├── extract_citations.py
│   ├── fetch_document.py
│   ├── clean_html.py
│   ├── create_passages.py
│   └── save_passages_to_db.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── nltk_setup.py       # NLTK tokenizer setup
│   └── progress.py         # Progress tracking
└── cache/                  # Runtime cache (created automatically)
```

---

## 🔧 Local Development Setup

If you need to develop or debug locally:

### 1. Install Python 3.14+

```bash
# Check Python version
python --version
```

### 2. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start PostgreSQL

```bash
docker compose up -d db
```

### 5. Run the pipeline

```bash
python main.py
```

---

## 🐳 Docker Commands Reference

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start all services in detached mode |
| `docker compose logs -f app` | Follow application logs |
| `docker compose logs -f db` | Follow database logs |
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop, remove containers and volumes |
| `docker compose build` | Rebuild the application image |
| `docker compose restart` | Restart all services |
| `docker compose ps` | List running containers |

---

## 🧪 Testing

```bash
# Run tests (if available)
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html
```

---

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker compose ps db

# View database logs
docker compose logs db

# Restart database
docker compose restart db
```

### Application Build Issues

```bash
# Rebuild the application image
docker compose build app

# Clean and rebuild
docker compose build --no-cache app
```

### NLTK Data Download Issues

The application automatically downloads NLTK data on first run. If you encounter issues:

```bash
# Manually download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

---

## 📊 Database Schema

The PostgreSQL database stores:

- **Articles**: Wikipedia article metadata
- **Citations**: References extracted from articles
- **Passages**: Cleaned text segments from cited documents
- **Documents**: Full text of cited documents

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is part of the Natural Language Processing course (Semester 2).

---

## 📞 Support

For issues or questions, please open an issue in the repository.

---

**Happy coding! 🚀**
"

