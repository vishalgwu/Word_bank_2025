# chat.py
import os
import numpy as np
import polars as pl
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.rag.summarize import generate_summaries
from src.rag.llm import generate_answer

load_dotenv()

st.set_page_config(page_title="AI Assistant – Word Bank", layout="wide")

st.title("🤖 Ask the Dashboard (AI Assistant)")
st.write(
    "Ask anything about wages, employment, industries, education, gender, or regions "
    "based on the Word Bank labour dataset."
)

DATA_PATH = "cleaned_data.parquet"


# --------------------------------------------------
# Build in-memory vector store (once, cached)
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_memory():
    # Load data
    if os.path.exists(DATA_PATH):
        df = pl.read_parquet(DATA_PATH)
    else:
        st.error(f"Parquet file '{DATA_PATH}' not found. Run the main app once to create it.")
        return None, None, None

    # Generate summaries
    st.info("Generating dataset summaries...")
    summaries = generate_summaries(df)
    texts = [s["text"] for s in summaries]

    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embeddings for all summaries
    st.info("Embedding summaries...")
    embeddings = model.encode(texts, convert_to_numpy=True)

    return model, texts, embeddings


model, texts, embeddings = build_memory()

if model is None:
    st.stop()

st.success("AI memory built from dataset summaries ✅")


# --------------------------------------------------
# Helper: retrieve top-k relevant summaries
# --------------------------------------------------
def retrieve_top_k(query: str, k: int = 5) -> list[str]:
    """Returns top-k chunks most semantically similar to user query."""
    q_vec = model.encode([query], convert_to_numpy=True)[0]

    # cosine similarity
    denom = (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-10)
    scores = (embeddings @ q_vec) / denom

    # indices of top-k scores
    top_idx = scores.argsort()[-k:][::-1]

    return [texts[i] for i in top_idx]


# --------------------------------------------------
# Chat UI
# --------------------------------------------------
query = st.text_input("Enter your question:")

ask_pressed = st.button("Ask")

if ask_pressed:
    if not query.strip():
        st.warning("❗ Please type a question first.")
    else:
        with st.spinner("🔍 Thinking with your dataset..."):
            chunks = retrieve_top_k(query, k=5)
            answer = generate_answer(query, chunks)

        st.subheader("📌 AI Answer")
        st.write(answer)
