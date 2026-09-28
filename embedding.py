"""
embedding.py
------------
This file handles loading the Sentence Transformer model, generating 
vector embeddings for text, calculating cosine similarity, and returning 
the top matching sentences.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads and returns a pre-trained Sentence Transformer model.

    Args:
        model_name (str): Name of the Hugging Face model. Defaults to 'all-MiniLM-L6-v2'.

    Returns:
        SentenceTransformer: Loaded embedding model instance.
    """
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model '{model_name}': {e}")


def generate_embeddings(sentences: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Generates numerical vector embeddings for a list of sentences.

    Args:
        sentences (list[str]): List of text sentences.
        model (SentenceTransformer): Loaded Sentence Transformer model.

    Returns:
        np.ndarray: 2D NumPy array of shape (num_sentences, embedding_dim).
    """
    if not sentences:
        return np.array([])
    
    # encode() converts text sentences into dense numerical vectors
    embeddings = model.encode(sentences, show_progress_bar=False)
    return np.array(embeddings)


def calculate_similarity(query_embedding: np.ndarray, dataset_embeddings: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity between query embedding and dataset embeddings.

    Args:
        query_embedding (np.ndarray): Vector representation of user query (1D or 2D).
        dataset_embeddings (np.ndarray): Matrix of embeddings for dataset sentences.

    Returns:
        np.ndarray: 1D NumPy array containing similarity scores (between -1.0 and 1.0).
    """
    # Ensure query embedding is a 2D array of shape (1, vector_dimension)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Compute pairwise cosine similarity between query and dataset matrix
    similarity_matrix = cosine_similarity(query_embedding, dataset_embeddings)
    
    # Extract 1D array of scores for the single query
    return similarity_matrix[0]


def search_similar_sentences(
    query: str,
    sentences: list[str],
    dataset_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5
) -> list[dict]:
    """
    Finds the top-K most semantically similar sentences for a given query.

    Args:
        query (str): User input search text.
        sentences (list[str]): Original dataset text sentences.
        dataset_embeddings (np.ndarray): Pre-calculated dataset embeddings matrix.
        model (SentenceTransformer): Sentence Transformer model instance.
        top_k (int): Number of top results to return. Defaults to 5.

    Returns:
        list[dict]: List of dictionaries containing 'rank', 'sentence', and 'score'.
    """
    if not query.strip():
        return []

    # 1. Convert user query to vector embedding
    query_embedding = model.encode([query])[0]

    # 2. Compute similarity score against all dataset sentences
    similarity_scores = calculate_similarity(query_embedding, dataset_embeddings)

    # 3. Sort indices in descending order (highest similarity first)
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # 4. Format Top-K results
    results = []
    top_n = min(top_k, len(sentences))
    
    for rank_idx in range(top_n):
        orig_idx = sorted_indices[rank_idx]
        results.append({
            "rank": rank_idx + 1,
            "sentence": sentences[orig_idx],
            "score": float(similarity_scores[orig_idx])
        })

    return results

