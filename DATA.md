# Dataset for LLM Book Recommender

Download one of the datasets below and place the CSV file(s) in this folder:

```
LLM Book Recommendor/
  data/
    raw/          <-- put your downloaded CSV here
```

---

## Recommended (≈7K books with metadata & descriptions)

**7K Books with Metadata**  
- **Link:** https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata  
- **Steps:** Sign in to Kaggle → Open the link → Click "Download" → Unzip and copy the CSV (e.g. `books.csv` or the main CSV) into `data/raw/`.

---

## Alternative (Goodreads-style, with descriptions)

**Books Dataset – Goodreads (May 2024)**  
- **Link:** https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024  
- Place the downloaded CSV in `data/raw/`.

**Goodreads Book Datasets With User Rating (large)**  
- **Link:** https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m  
- Use the CSV that contains book metadata/descriptions and place it in `data/raw/`. The pipeline will use the first ~6,810 rows (or you can pre-filter and save a smaller CSV).

---

## After downloading

- Put **any one** of the CSVs above (or any book CSV with columns like title, author, description) into **`data/raw/`**.
- The code looks for any `.csv` in `data/raw/` and supports common column names (title, author, description, genres, rating, etc.).
- Then run: `python data_prep.py` (and the rest of the pipeline as in the main README).
