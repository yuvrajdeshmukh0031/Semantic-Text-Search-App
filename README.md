# 🔎 Semantic Text Search using Embeddings

A beginner-friendly Python application that demonstrates how **text embeddings** and **semantic search** work using `sentence-transformers`, `scikit-learn`, `pypdf`, and `Streamlit`.

---

## 📌 Conceptual Guide

### What is an Embedding?
An **embedding** is a numerical representation of text (words, sentences, or paragraphs) in dense vector format. Instead of viewing words as plain text strings, machine learning models convert text into a long list of floating-point numbers (vectors). These numbers capture the **contextual meaning** and relationship of the words.

### What is Semantic Search?
Traditional keyword search looks for exact string matches (e.g., matching the word "car" with "car"). **Semantic search** understands the **meaning and intent** behind words. For example, a semantic search engine recognizes that *"automobile"* and *"car"* refer to the same concept even though they use completely different letters.

### What is Cosine Similarity?
**Cosine similarity** is a mathematical metric used to measure how similar two vectors are, regardless of their size. It measures the cosine of the angle between two vectors projected in a multidimensional vector space:
- **Score = 1.0**: Perfect semantic match (pointing in the exact same direction).
- **Score = 0.0**: Completely independent/unrelated meaning.
- **Score = -1.0**: Opposite meaning.

---

## 🏗️ How the Project Works

```
Text / PDF Document ──► Text Extraction ──► Sentence Transformer ──► 384-D Vector ──► Cosine Similarity ──► Ranked Results
```

1. **Dataset / PDF Upload (`data.py`)**: Stores sample sentences and provides PDF text extraction using `pypdf`.
2. **Embedding Generation (`embedding.py`)**: The pre-trained model `all-MiniLM-L6-v2` converts each sentence into a **384-dimensional vector**.
3. **Query Vectorization**: The user enters a search query which is converted into a vector in the same 384-D space.
4. **Similarity Calculation**: Cosine similarity measures the angle between the query vector and dataset vectors.
5. **Streamlit UI (`app.py`)**: Displays the Top 5 results ranked by similarity score with custom search scopes (PDF only, Sample Dataset only, or Combined).

---

## 📁 Project Structure

```
semantic-text-search/
│
├── app.py           # Streamlit Web Application Interface (with PDF Upload)
├── embedding.py     # SentenceTransformer & Cosine Similarity logic
├── data.py          # Sample dataset & pypdf extraction helper
├── requirements.txt # Python package dependencies
├── README.md        # Documentation and guide
└── .gitignore       # Git ignore rules
```

---

## ⚡ Setup & Installation

### 1. Prerequisites
Ensure you have Python **3.10 or higher** installed on your system.

### 2. Clone or Navigate to the Directory
```bash
cd semantic-text-search
```

### 3. Create a Virtual Environment (Optional but Recommended)
```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

Launch the Streamlit web application:

```bash
streamlit run app.py
```

After running the command, Streamlit will automatically open the application in your browser at `http://localhost:8501`.

---

## 💡 PDF Upload Feature

1. Open the sidebar in the Streamlit web interface.
2. Click **"📄 Upload a PDF File"** and choose any text-based PDF document.
3. The application will automatically extract sentences from the PDF and add them to the semantic search index!
4. Choose your preferred Search Scope:
   - **Combined (PDF + Default Data)**
   - **Uploaded PDF Only**
   - **Default Dataset Only**

---

## 🛠️ Built With

- **Python 3.10+**
- **Sentence-Transformers**: `all-MiniLM-L6-v2` pre-trained model
- **pypdf**: PDF document parsing & text extraction
- **NumPy**: Vector manipulation
- **scikit-learn**: Cosine similarity calculation
- **Streamlit**: Web interface
