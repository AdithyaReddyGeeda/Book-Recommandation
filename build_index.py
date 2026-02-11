"""
Build embedding index and emotion labels for the recommendation engine.
Reads processed books CSV, generates embeddings (sentence-transformers),
classifies emotion per book (DistilRoBERTa), and stores vectors in ChromaDB.
Run after data_prep.py. Safe for 16GB RAM with batched processing.
"""
import os
import pickle
import pandas as pd
import numpy as np

import config

# Lazy imports to avoid loading heavy models until needed
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL)


def _get_emotion_pipeline():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=config.EMOTION_MODEL,
        top_k=1,
        truncation=True,
        max_length=512,
    )


def _embed_texts(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts to vectors. For E5, prefix with 'passage: '."""
    if config.EMBEDDING_IS_E5:
        texts = [f"passage: {t}" if t else "passage: " for t in texts]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def add_emotion_labels(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Add emotion column using HuggingFace emotion model (7 classes)."""
    df = df.copy()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x
    pipe = _get_emotion_pipeline()
    descriptions = df["description"].fillna("").astype(str).tolist()
    descriptions = [d[:512] if d else "neutral" for d in descriptions]
    labels = []
    n_batches = (len(descriptions) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(descriptions), batch_size), total=n_batches, desc="Emotion batches", unit="batch"):
        batch = descriptions[i : i + batch_size]
        out = pipe(batch)
        for item in out:
            pred = item[0] if isinstance(item, list) else item
            label = pred.get("label", "neutral").lower().replace("_", " ")
            labels.append(label)
    df["emotion"] = labels
    return df


def build_chroma_index(processed_csv: str | None = None) -> pd.DataFrame:
    """
    Load processed books, compute embeddings and emotions, persist to ChromaDB.
    Returns DataFrame with all book metadata + emotion (saved to index dir for engine).
    """
    csv_path = processed_csv or config.PROCESSED_BOOKS_CSV
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Run data_prep.py first. Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"Loaded {n} books from {csv_path}")

    # Emotion labels (7 classes; may take 10–20 min for 6.8k books)
    print("Computing emotion labels...")
    df = add_emotion_labels(df)
    print("Emotion labels done.")

    # Embeddings
    print("Loading embedding model...")
    model = _get_embedding_model()
    texts = df["searchable_text"].fillna("").astype(str).tolist()
    print("Encoding book texts...")
    embeddings = _embed_texts(model, texts)
    print("Embeddings shape:", embeddings.shape)

    # ChromaDB
    import chromadb
    from chromadb.config import Settings

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_or_create_collection(
        config.CHROMA_COLLECTION_NAME,
        metadata={"description": "Book embeddings for recommendation"},
    )

    # Chroma expects id, embedding, and optional metadata (metadata size limited)
    ids = [str(i) for i in range(n)]
    def _str_slice(val, max_len: int):
        if pd.isna(val) or val is None:
            return ""
        return str(val)[:max_len]

    metadatas = [
        {
            "title": _str_slice(row.get("title"), 200),
            "author": _str_slice(row.get("author"), 200),
            "emotion": _str_slice(row.get("emotion"), 50),
            "rating": float(row.get("rating", 0)),
            "rating_count": int(row.get("rating_count", 0)),
        }
        for _, row in df.iterrows()
    ]
    # ChromaDB has a max batch size (~5461); add in chunks
    CHROMA_BATCH = 5000
    emb_list = embeddings.tolist()
    for start in range(0, n, CHROMA_BATCH):
        end = min(start + CHROMA_BATCH, n)
        collection.add(
            ids=ids[start:end],
            embeddings=emb_list[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Added {start}-{end} / {n}")
    print(f"ChromaDB updated: {config.CHROMA_PATH}")

    # Save full metadata for engine (description, image_url, etc.)
    meta_path = os.path.join(config.INDEX_DIR, "books_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(df, f)
    print(f"Metadata saved: {meta_path}")
    return df


if __name__ == "__main__":
    build_chroma_index()
