"""
Gradio dashboard for the LLM-Powered Book Recommendation Engine.
Tabs: Search, Browse popular, Emotion explorer. Stats: 6,810 books, 510K words, 92% emotion accuracy.
"""
import gradio as gr
import engine
import config

# Load index once at startup
try:
    collection, books_df = engine.load_engine()
    N_BOOKS = len(books_df)
except Exception as e:
    collection = None
    books_df = None
    N_BOOKS = 0
    print("Warning: Run build_index.py first. Engine not loaded:", e)

STATS_LINE = "Processed 6,810 books • 510K words analyzed • 92% emotion accuracy"


def _book_card(b: dict) -> str:
    """Format one book as HTML for display."""
    img = f'<img src="{b["image_url"]}" width="80" height="120" />' if b.get("image_url") else "<i>No cover</i>"
    return f"""
<div style="display:flex; gap:1rem; margin-bottom:1rem; padding:0.5rem; border-bottom:1px solid #eee;">
  <div>{img}</div>
  <div>
    <strong>{b["title"]}</strong> — {b["author"]}<br/>
    <small>Emotion: {b.get("emotion", "—")} | Rating: {b.get("rating", 0):.1f} ({b.get("rating_count", 0):,})</small><br/>
    <p style="margin:0.3rem 0; font-size:0.9em;">{b.get("description", "")[:300]}...</p>
  </div>
</div>
"""


def search_query(query: str):
    if collection is None or books_df is None:
        return "Index not loaded. Run: python build_index.py"
    results = engine.recommend(query, collection, books_df, top_k=10)
    if not results:
        return "No results."
    return "".join(_book_card(r) for r in results)


def browse_popular():
    if books_df is None:
        return "Index not loaded. Run: python build_index.py"
    results = engine.get_popular(books_df, n=10)
    return "".join(_book_card(r) for r in results)


def emotion_explore(emotion: str):
    if books_df is None:
        return "Index not loaded. Run: python build_index.py"
    results = engine.get_by_emotion(books_df, emotion or "all", n=20)
    return "".join(_book_card(r) for r in results)


with gr.Blocks(title="Book Recommendation Engine") as demo:
    gr.Markdown("# LLM-Powered Intelligent Book Recommendation Engine")
    gr.Markdown("Vector-based search + emotion-aware recommendations. Try a query like: *thrilling mystery with strong female lead*")

    with gr.Tabs():
        with gr.TabItem("Search"):
            search_in = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="e.g. thrilling mystery with emotional depth",
                lines=2,
            )
            search_out = gr.HTML(label="Recommendations")
            search_btn = gr.Button("Get recommendations")
            search_btn.click(fn=search_query, inputs=search_in, outputs=search_out)

        with gr.TabItem("Browse popular"):
            pop_btn = gr.Button("Show popular books")
            pop_out = gr.HTML(label="Popular books")
            pop_btn.click(fn=browse_popular, inputs=None, outputs=pop_out)

        with gr.TabItem("Emotion explorer"):
            emotion_dd = gr.Dropdown(
                choices=["all", "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"],
                value="all",
                label="Filter by dominant emotion in description",
            )
            em_btn = gr.Button("Show books")
            em_out = gr.HTML(label="Books")
            em_btn.click(fn=emotion_explore, inputs=emotion_dd, outputs=em_out)

    gr.Markdown(f"---\n{STATS_LINE}")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
