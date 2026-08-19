"""
rag/embedder.py

Generates embeddings for policy chunks using sentence-transformers
(local, hosted-free — no API key needed, fine for 6 short docs).

Model: paraphrase-multilingual-MiniLM-L12-v2 — supports Arabic (policy
content / findings may include Arabic, e.g. name_ar), same 384-dim
output as the English-only MiniLM so no downstream changes needed.
If retrieval quality on Arabic content isn't good enough, consider
BGE-M3 instead (used in the team's other Arabic RAG project) — it's
stronger on Arabic but ~2GB vs ~470MB, so budget setup time for it.
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model once per process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Take chunker output (list of chunk dicts) and return the same
    dicts with an added "embedding" key (list[float]).
    """
    model = get_model()
    texts = [f"{c['title']} — {c['section']}: {c['content']}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks


if __name__ == "__main__":
    from chunker import chunk_all_policies

    chunks = chunk_all_policies()
    embedded = embed_chunks(chunks)
    print(f"Embedded {len(embedded)} chunks, dim={len(embedded[0]['embedding'])}")