"""
Data preparation for the Book Recommendation Engine.
Loads raw book CSV, cleans text, concatenates searchable content, and computes
total word count (~510K words across descriptions/metadata) for portfolio metrics.
"""
import os
import re
import pandas as pd
import numpy as np

import config

# -----------------------------------------------------------------------------
# Column mapping: support multiple CSV schemas (Kaggle 7k, Goodreads, BX-style)
# -----------------------------------------------------------------------------
POSSIBLE_TITLE_COLS = ["title", "Title", "book_title", "name"]
POSSIBLE_AUTHOR_COLS = ["author", "Author", "authors", "writer"]
POSSIBLE_DESC_COLS = ["description", "Description", "desc", "summary", "Summary", "synopsis"]
POSSIBLE_GENRE_COLS = ["genres", "genre", "Genres", "categories"]
POSSIBLE_RATING_COLS = ["rating", "Rating", "average_rating", "avg_rating"]
POSSIBLE_RATING_COUNT_COLS = ["rating_count", "ratings_count", "Rating Count", "num_ratings"]
POSSIBLE_IMAGE_COLS = ["image_url", "thumbnail", "cover", "img", "image"]


def _find_column(df: pd.DataFrame, candidates: list) -> str | None:
    """Return first column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various CSV column names to a standard schema."""
    out = pd.DataFrame(index=df.index)
    out["title"] = df[_find_column(df, POSSIBLE_TITLE_COLS) or "title"].astype(str)
    out["author"] = df[_find_column(df, POSSIBLE_AUTHOR_COLS) or "author"].astype(str)
    desc_col = _find_column(df, POSSIBLE_DESC_COLS)
    out["description"] = df[desc_col].astype(str) if desc_col else ""
    genre_col = _find_column(df, POSSIBLE_GENRE_COLS)
    out["genres"] = df[genre_col].astype(str) if genre_col else ""
    rating_col = _find_column(df, POSSIBLE_RATING_COLS)
    out["rating"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0.0) if rating_col else 0.0
    rc_col = _find_column(df, POSSIBLE_RATING_COUNT_COLS)
    out["rating_count"] = pd.to_numeric(df[rc_col], errors="coerce").fillna(0).astype(int) if rc_col else 0
    img_col = _find_column(df, POSSIBLE_IMAGE_COLS)
    out["image_url"] = df[img_col].astype(str) if img_col else ""
    return out


def clean_text(s: str) -> str:
    """Normalize and clean a string for embedding."""
    if pd.isna(s) or not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s[:10000]  # cap length for memory


def build_searchable_text(row: pd.Series) -> str:
    """Concatenate title, author, description, genres for embedding."""
    parts = [
        clean_text(row["title"]),
        clean_text(row["author"]),
        clean_text(row["description"]),
        clean_text(row["genres"]),
    ]
    return " | ".join(p for p in parts if p)


def load_raw_books(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load raw books from CSV. If csv_path is None, looks in data/raw for
    books.csv, books_processed.csv, or any *.csv.
    """
    if csv_path and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, nrows=7000, low_memory=False)
        return normalize_columns(df)

    raw_dir = config.RAW_DATA_DIR
    for name in ("books.csv", "books_processed.csv", "goodreads.csv", "7k_books.csv"):
        path = os.path.join(raw_dir, name)
        if os.path.isfile(path):
            df = pd.read_csv(path, nrows=7000, low_memory=False)
            return normalize_columns(df)

    # Check any CSV in raw
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(".csv"):
                path = os.path.join(raw_dir, f)
                df = pd.read_csv(path, nrows=7000, low_memory=False)
                return normalize_columns(df)

    # Fallback: project root (e.g. books.csv next to data_prep.py)
    for name in ("books.csv", "books_processed.csv", "goodreads.csv"):
        path = os.path.join(config.PROJECT_ROOT, name)
        if os.path.isfile(path):
            df = pd.read_csv(path, nrows=7000, low_memory=False)
            return normalize_columns(df)

    raise FileNotFoundError(
        f"No CSV found in {raw_dir} or project root. "
        "Please add a books CSV (e.g. from Kaggle '7k-books-with-metadata') to data/raw/ or project root."
    )


def prepare_books(
    csv_path: str | None = None,
    target_count: int = 6810,
    min_description_length: int = 10,
) -> pd.DataFrame:
    """
    Load, clean, and prepare books. Ensures we have ~target_count books with
    usable descriptions where possible. Outputs processed CSV and returns DataFrame.
    """
    df = load_raw_books(csv_path)

    # Prefer rows that have some description
    df["description"] = df["description"].apply(clean_text)
    has_desc = df["description"].str.len() >= min_description_length
    if has_desc.sum() >= target_count:
        df = df.loc[has_desc].head(target_count).copy()
    else:
        df = df.head(target_count).copy()

    df["searchable_text"] = df.apply(build_searchable_text, axis=1)
    df["word_count"] = df["searchable_text"].str.split().str.len().fillna(0).astype(int)
    total_words = int(df["word_count"].sum())

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.to_csv(config.PROCESSED_BOOKS_CSV, index=False)
    print(f"Processed {len(df)} books, total words ≈ {total_words:,}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to raw books CSV")
    parser.add_argument("--target", type=int, default=6810, help="Target number of books")
    args = parser.parse_args()
    prepare_books(csv_path=args.csv, target_count=args.target)
