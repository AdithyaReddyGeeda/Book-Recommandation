"""
Recommendation engine: vector search + sales/popularity scoring.
Uses ChromaDB for similarity search and optional reranking by popularity.
"""
import os
import pickle
import numpy as np
import pandas as pd

import config

# Lazy load embedding model
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


def _embed_query(model, query: str) -> np.ndarray:
    if config.EMBEDDING_IS_E5:
        query = f"query: {query}"
    return model.encode([query], show_progress_bar=False)[0]


def sales_score(row: pd.Series) -> float:
    """Popularity/sellability: rating * log(1 + rating_count) + small emotion bonus."""
    r = float(row.get("rating", 0))
    rc = int(row.get("rating_count", 0))
    base = r * np.log1p(rc)
    return base + config.SENTIMENT_BONUS


def load_engine():
    """Load Chroma collection and books metadata. Call once at app startup."""
    import chromadb

    if not os.path.isdir(config.CHROMA_PATH):
        raise FileNotFoundError(
            f"Index not found at {config.CHROMA_PATH}. Run: python build_index.py"
        )
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_collection(config.CHROMA_COLLECTION_NAME)

    meta_path = os.path.join(config.INDEX_DIR, "books_metadata.pkl")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}. Run build_index.py")
    with open(meta_path, "rb") as f:
        books_df = pickle.load(f)

    return collection, books_df


def recommend(
    query: str,
    collection,
    books_df: pd.DataFrame,
    top_k: int = None,
    retrieval_k: int = None,
) -> list[dict]:
    """
    Recommend books by semantic similarity, then optionally boost by sales score.
    Returns list of dicts with title, author, description, image_url, emotion, score, etc.
    """
    top_k = top_k or config.TOP_K_RETURN
    retrieval_k = retrieval_k or config.TOP_K_RETRIEVAL

    if not query or not query.strip():
        # No query: return popular
        return get_popular(books_df, n=top_k)

    model = _get_embedding_model()
    q_embed = _embed_query(model, query.strip()).tolist()

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=min(retrieval_k, collection.count()),
        include=["metadatas", "distances"],
    )
    ids = [int(x) for x in results["ids"][0]]
    distances = results["distances"][0]

    # Chroma returns L2 distance; convert to similarity-like (lower distance = higher score)
    sim_scores = 1.0 / (1.0 + np.array(distances))
    rows = books_df.iloc[ids].copy()
    rows["similarity"] = sim_scores
    rows["sales"] = rows.apply(sales_score, axis=1)
    # Combined score: similarity + sales boost (normalize sales to ~0-0.5 range)
    max_sales = rows["sales"].max() or 1
    rows["score"] = rows["similarity"] + 0.3 * (rows["sales"] / max_sales)
    rows = rows.sort_values("score", ascending=False).head(top_k)

    out = []
    for _, r in rows.iterrows():
        out.append({
            "title": r["title"],
            "author": r["author"],
            "description": (r.get("description") or "")[:500],
            "image_url": r.get("image_url") or "",
            "emotion": r.get("emotion", ""),
            "rating": r.get("rating", 0),
            "rating_count": int(r.get("rating_count", 0)),
            "score": float(r["score"]),
        })
    return out


def get_popular(books_df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Top N books by sales/popularity score."""
    df = books_df.copy()
    df["sales"] = df.apply(sales_score, axis=1)
    df = df.sort_values("sales", ascending=False).head(n)
    return [
        {
            "title": r["title"],
            "author": r["author"],
            "description": (r.get("description") or "")[:500],
            "image_url": r.get("image_url") or "",
            "emotion": r.get("emotion", ""),
            "rating": r.get("rating", 0),
            "rating_count": int(r.get("rating_count", 0)),
            "score": float(r["sales"]),
        }
        for _, r in df.iterrows()
    ]


def get_by_emotion(books_df: pd.DataFrame, emotion: str, n: int = 20) -> list[dict]:
    """Books tagged with given emotion (e.g. joy, sadness)."""
    if not emotion or emotion.strip().lower() == "all":
        return get_popular(books_df, n=n)
    em = emotion.strip().lower()
    df = books_df[books_df["emotion"].str.lower().str.contains(em, na=False)].copy()
    df["sales"] = df.apply(sales_score, axis=1)
    df = df.sort_values("sales", ascending=False).head(n)
    return [
        {
            "title": r["title"],
            "author": r["author"],
            "description": (r.get("description") or "")[:500],
            "image_url": r.get("image_url") or "",
            "emotion": r.get("emotion", ""),
            "rating": r.get("rating", 0),
            "rating_count": int(r.get("rating_count", 0)),
            "score": float(r["sales"]),
        }
        for _, r in df.iterrows()
    ]
