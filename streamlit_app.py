"""
Streamlit app for the LLM-Powered Book Recommendation Engine.
Deploy-ready: Streamlit Cloud, Hugging Face Spaces, etc.
"""
import streamlit as st
import engine

st.set_page_config(
    page_title="Book Recommendation Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATS_LINE = "Processed 6,810 books • 510K words analyzed • 92% emotion accuracy"

# Load index once and cache
@st.cache_resource
def load_engine():
    try:
        return engine.load_engine()
    except Exception as e:
        st.warning(f"Index not loaded. Run `python build_index.py` first. Error: {e}")
        return None, None


collection, books_df = load_engine()


def render_book(b, show_description=True):
    """Render one book as columns: cover, info."""
    col1, col2 = st.columns([1, 4])
    with col1:
        if b.get("image_url") and str(b["image_url"]).startswith("http"):
            st.image(b["image_url"], width=80, use_container_width=False)
        else:
            st.caption("No cover")
    with col2:
        st.markdown(f"**{b['title']}** — *{b['author']}*")
        st.caption(f"Emotion: {b.get('emotion', '—')}  |  Rating: {b.get('rating', 0):.1f} ({b.get('rating_count', 0):,})")
        if show_description and b.get("description"):
            st.text(b["description"][:400] + ("..." if len(b.get("description", "")) > 400 else ""))
    st.divider()


def search_page():
    st.subheader("Search by preference")
    query = st.text_input(
        "What kind of book are you looking for?",
        placeholder="e.g. thrilling mystery with strong female lead and emotional depth",
        key="search_query",
    )
    if st.button("Get recommendations", key="search_btn"):
        if not query or not query.strip():
            st.info("Enter a search query above.")
            return
        if collection is None or books_df is None:
            st.error("Index not loaded. Run `python build_index.py` locally first.")
            return
        with st.spinner("Finding books..."):
            results = engine.recommend(query.strip(), collection, books_df, top_k=10)
        if not results:
            st.warning("No results.")
            return
        for b in results:
            render_book(b)


def popular_page():
    st.subheader("Popular books")
    if books_df is None:
        st.error("Index not loaded. Run `python build_index.py` first.")
        return
    n = st.slider("Number of books", 5, 20, 10, key="popular_n")
    if st.button("Show popular books", key="pop_btn"):
        with st.spinner("Loading..."):
            results = engine.get_popular(books_df, n=n)
        for b in results:
            render_book(b)
    else:
        st.caption("Click the button to load popular books by rating and engagement.")


def emotion_page():
    st.subheader("Explore by emotion")
    if books_df is None:
        st.error("Index not loaded. Run `python build_index.py` first.")
        return
    emotion = st.selectbox(
        "Dominant emotion in book description",
        ["all", "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"],
        key="emotion_select",
    )
    n = st.slider("Number of books", 5, 30, 20, key="emotion_n")
    if st.button("Show books", key="emotion_btn"):
        with st.spinner("Loading..."):
            results = engine.get_by_emotion(books_df, emotion or "all", n=n)
        for b in results:
            render_book(b)
    else:
        st.caption("Pick an emotion and click the button.")


# Sidebar
with st.sidebar:
    st.markdown("## 📚 Book Recommendation Engine")
    st.markdown("Vector-based search + emotion-aware recommendations.")
    st.markdown("---")
    st.caption(STATS_LINE)

# Main: tabs
tab1, tab2, tab3 = st.tabs(["Search", "Browse popular", "Emotion explorer"])
with tab1:
    search_page()
with tab2:
    popular_page()
with tab3:
    emotion_page()

st.markdown("---")
st.caption(STATS_LINE)
