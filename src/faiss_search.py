import faiss
import numpy as np
import os
import pickle

# Path where FAISS index will be saved
INDEX_PATH = "models/faiss_index.bin"


def build_faiss_index(embeddings):
    """
    Build FAISS index from embeddings
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    return index


def save_index(index):
    """
    Save FAISS index to disk
    """

    faiss.write_index(index, INDEX_PATH)


def load_index():
    """
    Load FAISS index
    """

    if not os.path.exists(INDEX_PATH):
        raise ValueError("FAISS index not found")

    return faiss.read_index(INDEX_PATH)


def search_similar(index, query_embedding, k=5):
    """
    Search for similar embeddings
    """

    distances, indices = index.search(query_embedding.astype("float32"), k)

    return indices, distances