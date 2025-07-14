import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)

__all__ = ["init", "answer_question", "_text_chunks", "_matrix", "get_chunks"]

_text_chunks: list[str] = []
_vectorizer = TfidfVectorizer(stop_words="english")
_matrix = None


def init(doc_text: str, chunk_size: int = 150):
    """
    Split the document into fixed-size word chunks and compute TF-IDF matrix.
    """
    global _text_chunks, _matrix
    words = doc_text.split()
    _text_chunks = [
        " ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    _matrix = _vectorizer.fit_transform(_text_chunks)
    print(f"[✅] Indexed {len(_text_chunks)} chunks. TF-IDF shape: {_matrix.shape}")


def answer_question(question: str) -> tuple[str, str]:
    """
    Find the most relevant document chunk for the given question.
    Returns (answer_text, reference_tag)
    """
    if _matrix is None:
        return "Document not indexed yet.", ""

    q_vec = _vectorizer.transform([question])
    sims = cosine_similarity(q_vec, _matrix).flatten()
    best_idx = sims.argmax()
    return _text_chunks[best_idx], f"📍 Source: Chunk #{best_idx}"


def get_chunks() -> list[str]:
    """
    Utility: Return current document chunks for debugging/testing.
    """
    return _text_chunks
