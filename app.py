"""
app.py
------
Streamlit User Interface for Semantic Text Search using Embeddings.
Allows users to enter a natural language query and retrieve the top 5 
most semantically similar sentences from the sample dataset OR an uploaded PDF document.
"""

import streamlit as st
import numpy as np

from data import get_sample_sentences, extract_sentences_from_pdf
from embedding import load_model, generate_embeddings, search_similar_sentences


# Configure Streamlit Page Settings
st.set_page_config(
    page_title="Semantic Text Search",
    page_icon="🔎",
    layout="wide"
)

# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------

@st.cache_resource
def get_cached_model():
    """
    Loads and caches the SentenceTransformer model to prevent reloading
    on every user interaction.
    """
    return load_model("all-MiniLM-L6-v2")


def get_dynamic_embeddings(model, sentences: list[str]):
    """
    Generates embeddings dynamically for the provided sentence list.
    """
    return generate_embeddings(sentences, model)


# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------

def main():
    # Title & Subheader
    st.title("🔎 Semantic Text Search")
    st.markdown(
        "Search text based on **meaning and intent**, not just exact keyword matches!"
    )
    st.write("---")

    # Load default sample dataset
    sample_sentences = get_sample_sentences()
    
    with st.spinner("Loading AI Embedding Model (`all-MiniLM-L6-v2`)..."):
        model = get_cached_model()

    # ---------------------------------------------------------
    # Sidebar: PDF Upload & Dataset Selection
    # ---------------------------------------------------------
    st.sidebar.header("📁 Data Source & Options")

    # PDF File Uploader
    uploaded_pdf = st.sidebar.file_uploader(
        "📄 Upload a PDF File",
        type=["pdf"],
        help="Upload any text-based PDF to extract its sentences for semantic search."
    )

    pdf_sentences = []
    if uploaded_pdf is not None:
        with st.sidebar.spinner("Extracting text from PDF..."):
            pdf_sentences = extract_sentences_from_pdf(uploaded_pdf)
        
        if pdf_sentences:
            st.sidebar.success(f"✅ Extracted **{len(pdf_sentences)}** sentences from `{uploaded_pdf.name}`")
        else:
            st.sidebar.error("Could not extract readable text from this PDF.")

    # Select Search Scope
    if pdf_sentences:
        search_mode = st.sidebar.radio(
            "Select Search Scope:",
            options=["Combined (PDF + Default Data)", "Uploaded PDF Only", "Default Dataset Only"],
            index=0
        )
        
        if search_mode == "Uploaded PDF Only":
            active_sentences = pdf_sentences
        elif search_mode == "Default Dataset Only":
            active_sentences = sample_sentences
        else:
            active_sentences = pdf_sentences + sample_sentences
    else:
        active_sentences = sample_sentences

    # Generate Embeddings for Active Dataset
    dataset_embeddings = get_dynamic_embeddings(model, active_sentences)

    # Sidebar Dataset Metrics
    st.sidebar.write("---")
    st.sidebar.header("📊 Dataset Overview")
    st.sidebar.info(f"**Active Sentences:** {len(active_sentences)}")
    
    # Display Embedding Vector Dimension
    embedding_dim = dataset_embeddings.shape[1] if dataset_embeddings.ndim > 1 and len(active_sentences) > 0 else 384
    st.sidebar.metric(label="Embedding Dimension", value=f"{embedding_dim} numbers")
    
    with st.sidebar.expander("📄 View Active Sentence Index"):
        for i, s in enumerate(active_sentences, 1):
            st.write(f"**{i}.** {s}")

    # ---------------------------------------------------------
    # Main Query & Search Section
    # ---------------------------------------------------------
    st.subheader("🔍 Enter Your Search Query")
    
    # Text Input Box
    query = st.text_input(
        label="Enter your query",
        placeholder="e.g., I want to learn about artificial intelligence",
        value="I want to learn about artificial intelligence"
    )

    # Search Button
    search_button = st.button("Search", type="primary")

    if (search_button or query) and len(active_sentences) > 0:
        if not query.strip():
            st.warning("Please enter a non-empty search query.")
        else:
            # Perform Semantic Search
            results = search_similar_sentences(
                query=query,
                sentences=active_sentences,
                dataset_embeddings=dataset_embeddings,
                model=model,
                top_k=5
            )

            # Display Search Results Header
            st.write("---")
            st.subheader("🎯 Top 5 Most Similar Results")
            st.caption("💡 *Higher similarity score means the meaning of the two texts is more similar.*")

            # Display Results in structured cards
            for item in results:
                col_rank, col_content, col_score = st.columns([1, 6, 2])
                
                with col_rank:
                    st.subheader(f"#{item['rank']}")
                    
                with col_content:
                    st.write(f"**{item['sentence']}**")
                    
                with col_score:
                    score = item['score']
                    st.metric(label="Similarity Score", value=f"{score:.4f}")
                    st.progress(max(0.0, min(1.0, score)))
                
                st.divider()

    # Educational Section: How Embeddings Work
    st.write("---")
    st.subheader("💡 How Embeddings Work")
    st.markdown("""
    This project demonstrates how semantic search works step-by-step:
    
    ```
    Text / PDF Document ──► Text Extraction ──► Embedding Model ──► Numerical Vector ──► Cosine Similarity ──► Top Results
    ```
    
    1. **Text / PDF Input**: You enter a text query or upload a PDF document.
    2. **Text Extraction**: PyPDF parses the uploaded PDF file page-by-page into clean sentences.
    3. **Embedding**: The `all-MiniLM-L6-v2` transformer model converts text into a mathematical representation.
    4. **Vector**: Each sentence becomes a **384-dimensional vector** of floating-point numbers containing semantic meaning.
    5. **Similarity Search**: Cosine similarity compares the angle between the query vector and dataset vectors in 384-D space.
    6. **Result**: Sentences with the highest cosine similarity scores are ranked at the top!
    """)


if __name__ == "__main__":
    main()
