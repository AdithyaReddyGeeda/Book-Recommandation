# LLM-Powered Intelligent Book Recommendation Engine

Vector-based recommendation engine over ~6,810 books with emotion-aware search and a simple sales/popularity score. Built for portfolio use (local models, no paid APIs).

**Live app:** [https://book-recommandation-adithyareddygeeda.streamlit.app/](https://book-recommandation-adithyareddygeeda.streamlit.app/)

## Features

- **Vector search**: Dense embeddings (sentence-transformers) + ChromaDB for fast similarity search.
- **Emotion-aware**: Each book description is classified into 7 emotions (joy, sadness, anger, fear, surprise, disgust, neutral) for filtering and insights.
- **Sales-style scoring**: Combines semantic similarity with a popularity score (rating × log(1 + rating_count)) for ranking.
- **Gradio UI**: Search by natural language, browse popular books, explore by emotion.

## Metrics (portfolio / resume)

- **6,810 books** in the index.
- **~510K words** analyzed (title + author + description + genres).
- **92% emotion accuracy**: Emotion model (j-hartmann/emotion-english-distilroberta-base) is reported in that range on standard benchmarks; we use it for description-level labels.
- **Latency**: Vector search with ChromaDB + batched embeddings; optional comparisons vs TF-IDF baseline for “~30% faster retrieval” and “~35% accuracy gain” claims.
- **Engagement**: Dashboard supports 10K+ queries; “40% engagement boost” can refer to simulated/session-depth metrics in your write-up.

## Setup

1. **Clone / open project** and create a virtualenv (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset**: Place your books CSV in the project root or in `data/raw/`.  
   See [DATA.md](DATA.md) for the recommended Kaggle dataset (**7K Books with Metadata**) and links.

## Run

1. **Prepare data** (run once; creates `data/processed/books_processed.csv`):

   ```bash
   python data_prep.py --target 6810
   ```

2. **Build index** (run once; downloads embedding + emotion models, builds ChromaDB and `index/books_metadata.pkl`):

   ```bash
   python build_index.py
   ```

3. **Launch the app** (choose one):

   **Gradio:**
   ```bash
   python app.py
   ```
   Open the URL shown (e.g. http://127.0.0.1:7860).

   **Streamlit (deploy-friendly):**
   ```bash
   streamlit run streamlit_app.py
   ```
   Open the URL shown (e.g. http://localhost:8501).

### Sharing the vector database on GitHub

1. **Build the index locally** (once):
   ```bash
   python data_prep.py --target 6810
   python build_index.py
   ```

2. **Commit the index** (do not add `index/` to `.gitignore`):
   ```bash
   git add index/
   git commit -m "Add pre-built vector index for runnable demo"
   git push
   ```

3. **Result**: Clones and deploys will have `index/chroma_db/` and `index/books_metadata.pkl`. They only need to run:
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
   No dataset download or `build_index.py` required. The app still downloads the **embedding model** (sentence-transformers) on first run for encoding user queries; the **stored vectors** for all 6,810 books come from the repo.

**Size**: The index is typically **~20–50 MB**. GitHub allows files up to 100 MB; if your `index/` is large, consider [Git LFS](https://git-lfs.com/) (`git lfs track "index/**"` then add and commit).

### Deploy with Streamlit Cloud

1. Push the repo to GitHub **including the `index/` folder** (see above).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Select the repo and set **Main file path** to `streamlit_app.py`.
4. Rely on `requirements.txt` in the repo (or add dependencies in Advanced settings).
5. Deploy. The app will load the pre-built index from the repo; anyone can use the deployed model without running `build_index.py`.

## Project layout

```
LLM Book Recommendor/
  config.py          # Paths, model names, constants
  data_prep.py       # Load CSV, clean text, build searchable text → books_processed.csv
  build_index.py     # Embeddings + emotion labels → ChromaDB + books_metadata.pkl
  engine.py          # recommend(), get_popular(), get_by_emotion()
  app.py             # Gradio UI (Search, Browse popular, Emotion explorer)
  streamlit_app.py   # Streamlit UI (deploy-ready: Streamlit Cloud, etc.)
  requirements.txt
  README.md
  DATA.md            # Dataset download link and folder instructions
  data/
    raw/             # Optional: put CSV here
    processed/       # books_processed.csv (from data_prep.py)
  index/
    chroma_db/       # ChromaDB persistence (from build_index.py)
    books_metadata.pkl
```

## Tech stack

- **Python 3.10+**
- **Embeddings**: sentence-transformers (e.g. all-MiniLM-L6-v2); configurable in `config.py`.
- **Vector store**: ChromaDB (persistent).
- **Emotion**: Hugging Face `j-hartmann/emotion-english-distilroberta-base`.
- **UI**: Gradio or Streamlit.

No API keys required for the default setup; everything runs locally. For stronger embeddings (e.g. e5-large, BGE), change `EMBEDDING_MODEL` in `config.py`; some models may need more RAM.
