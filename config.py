"""
config.py — EvoRAG central configuration

All tunable constants live here so every phase imports from one place.
"""
# ── Phase 2 — Embedding + Vector Store (added below Phase 1 constants)

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# How many chunk JSON files to keep in data/ before rotating (deleting oldest)
MAX_CHUNK_FILES = 5

# ── Fetcher settings ──────────────────────────────────────────────────────────
MAX_ARTICLES_PER_SOURCE = 10   # articles to request from each source per query

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_TARGET_WORDS   = 350    # aim for this word count per chunk
CHUNK_MIN_WORDS      = 300    # discard chunks shorter than this
CHUNK_MAX_WORDS      = 400    # hard ceiling (split further if exceeded)
CHUNK_OVERLAP_WORDS  = 50     # overlap between consecutive chunks

# ── RSS feed list ─────────────────────────────────────────────────────────────
RSS_FEEDS = [
    {"name": "BBC World",    "url": "http://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "Reuters",      "url": "https://feeds.reuters.com/reuters/topNews"},
    {"name": "Al Jazeera",   "url": "https://www.aljazeera.com/xml/rss/all.xml"},
]

# ── Phase 2 — Embedding + Vector Store ───────────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR        = ROOT_DIR / "index"          # persisted FAISS files live here
FAISS_INDEX_FILE = INDEX_DIR / "evorag.faiss"
META_FILE        = INDEX_DIR / "evorag_meta.json"
EMBED_BATCH_SIZE = 64                          # chunks embedded per batch
TOP_K_DEFAULT    = 5                           # default retrieval count

# ── Phase 3 — Retrieval Pipeline + FastAPI ────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"   # default Ollama address
OLLAMA_MODEL     = "phi3:latest"              # confirmed model
API_HOST         = "0.0.0.0"
API_PORT         = 8000
RETRIEVAL_TOP_K  = 5                          # chunks fed into prompt
