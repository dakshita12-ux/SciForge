import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = ROOT_DIR / "knowledge"

DATA_DIR = ROOT_DIR / "data"

INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 120
CHUNK_OVERLAP = 30


def load_embedding_model():
    """Load the sentence embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def split_into_chunks(text):
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    document_paths = sorted(KNOWLEDGE_DIR.rglob("*.txt"))

    if not document_paths:
        raise RuntimeError(
            "No .txt documents found in the knowledge folder."
        )

    texts = []
    metadata = []

    for document_path in document_paths:
        document_text = document_path.read_text(encoding="utf-8")

        chunks = split_into_chunks(document_text)

        for chunk_number, chunk in enumerate(chunks):
            texts.append(chunk)

            metadata.append(
                {
                    "source": str(
                        document_path.relative_to(ROOT_DIR)
                    ).replace("\\", "/"),
                    "chunk_id": chunk_number,
                    "text": chunk,
                }
            )

    if not texts:
        raise RuntimeError(
            "No usable text chunks were generated."
        )

    print(f"Documents found: {len(document_paths)}")
    print(f"Chunks created: {len(texts)}")

    embedding_model = load_embedding_model()

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype("float32")

    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"FAISS index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()