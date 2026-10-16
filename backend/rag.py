import json
from functools import lru_cache
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = ROOT_DIR / "data" / "faiss.index"
METADATA_PATH = ROOT_DIR / "data" / "metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def load_vector_store():
    """Load the saved FAISS index and its metadata."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Knowledge index not found. Run: python -m backend.ingest"
        )

    index = faiss.read_index(str(INDEX_PATH))

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if index.ntotal != len(metadata):
        raise RuntimeError(
            "Index and metadata do not match. Rebuild the index."
        )

    if index.ntotal == 0:
        raise RuntimeError(
            "The knowledge index is empty. Add documents and rebuild it."
        )

    return index, metadata


def retrieve_documents(query, top_k=4):
    """
    Return the most relevant document chunks for a question.
    """
    query = query.strip()

    if not query:
        return []

    index, metadata = load_vector_store()
    embedding_model = load_embedding_model()

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    result_count = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, result_count)

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue

        result = metadata[int(index_position)].copy()
        result["score"] = float(score)

        results.append(result)

    return results