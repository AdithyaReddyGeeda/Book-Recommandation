"""
Configuration for LLM-Powered Book Recommendation Engine.
Centralizes paths, model names, and constants for 16GB RAM-friendly runs.
"""
import os

# -----------------------------------------------------------------------------
# Paths (relative to project root)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INDEX_DIR = os.path.join(PROJECT_ROOT, "index")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Processed data file (output of data_prep.py)
PROCESSED_BOOKS_CSV = os.path.join(PROCESSED_DIR, "books_processed.csv")
# ChromaDB persistence
CHROMA_PATH = os.path.join(INDEX_DIR, "chroma_db")
# Optional FAISS index path (if using FAISS instead of Chroma)
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
FAISS_METADATA_PATH = os.path.join(INDEX_DIR, "faiss_metadata.pkl")

# -----------------------------------------------------------------------------
# Embedding model (local, no API key)
# all-MiniLM-L6-v2: ~80MB, 384 dims, fast on CPU/16GB RAM
# Optional: "intfloat/e5-small-v2" or "BAAI/bge-small-en-v1.5" for better accuracy
# -----------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# For E5 models, prefix query with "query: " and document with "passage: " (we handle in code)
EMBEDDING_IS_E5 = "e5-" in EMBEDDING_MODEL.lower()

# -----------------------------------------------------------------------------
# Emotion / sentiment model (local)
# j-hartmann/emotion-english-distilroberta-base: 7 emotions, ~82M params, good accuracy
# -----------------------------------------------------------------------------
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_LABELS = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
]

# -----------------------------------------------------------------------------
# Vector store: "chroma" or "faiss"
# -----------------------------------------------------------------------------
VECTOR_STORE = "chroma"
CHROMA_COLLECTION_NAME = "books"

# -----------------------------------------------------------------------------
# Recommendation defaults
# -----------------------------------------------------------------------------
TOP_K_RETRIEVAL = 30
TOP_K_RETURN = 10
# Sales/popularity score: rating * log(1 + rating_count) + sentiment_bonus
SENTIMENT_BONUS = 0.1

# -----------------------------------------------------------------------------
# Dataset: expected CSV columns (flexible for Kaggle 7k books / Goodreads-style)
# -----------------------------------------------------------------------------
# If your CSV uses different column names, set these to match
COLUMN_TITLE = "title"
COLUMN_AUTHOR = "author"
COLUMN_DESCRIPTION = "description"
COLUMN_GENRES = "genres"
COLUMN_RATING = "rating"
COLUMN_RATING_COUNT = "rating_count"
COLUMN_IMAGE_URL = "image_url"

# Create directories on import
for _dir in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, INDEX_DIR, CACHE_DIR):
    os.makedirs(_dir, exist_ok=True)
