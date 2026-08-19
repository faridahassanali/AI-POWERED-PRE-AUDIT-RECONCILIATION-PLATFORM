"""
rag/vector_store.py

Loads embedded policy chunks into Qdrant and exposes a retrieval
function. Requires a Qdrant server running (see docker command below).

    docker run -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/rag/.qdrant_storage:/qdrant/storage \
        qdrant/qdrant

This module is intentionally the ONLY place that talks to Qdrant.
Step 3 (AI Input Contract) and Step 6 (Real LLM Integration) should
call retrieve_policy_context(), not touch the Qdrant client directly.
"""

import hashlib
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

_COLLECTION_NAME = "policy_chunks"
_QDRANT_HOST = "localhost"
_QDRANT_PORT = 6333
_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size — update if you change the model

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if _COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=_COLLECTION_NAME,
            vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
        )


def _chunk_id(policy_id: str, section: str) -> int:
    """Qdrant point IDs must be int or UUID — derive a stable int from policy_id+section."""
    key = f"{policy_id}::{section}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def index_chunks(chunks: List[Dict]) -> None:
    """
    Load embedded chunks (each must already have an "embedding" key)
    into the Qdrant collection. Re-indexing overwrites points with the
    same policy_id+section id, so this is safe to re-run after policy edits.
    """
    client = get_client()
    _ensure_collection(client)

    points = [
        PointStruct(
            id=_chunk_id(c["policy_id"], c["section"]),
            vector=c["embedding"],
            payload={
                "policy_id": c["policy_id"],
                "version": c["version"],
                "title": c["title"],
                "section": c["section"],
                "content": c["content"],
                "source_file": c["source_file"],
            },
        )
        for c in chunks
    ]

    client.upsert(collection_name=_COLLECTION_NAME, points=points)


def retrieve_policy_context(query: str, top_k: int = 3) -> List[Dict]:
    """
    Retrieve the top_k most relevant policy chunks for a query string
    (e.g. a finding's summary + control_id). Returns chunk-shaped dicts
    (policy_id, section, content, ...) — NOT raw Qdrant output — so
    downstream code (Step 3 AI Input Contract) doesn't need to know
    Qdrant's response shape.
    """
    try:
        from RAG.embedder import get_model
    except ImportError:
        from embedder import get_model

    model = get_model()
    query_embedding = model.encode([query])[0].tolist()

    client = get_client()
    results = client.query_points(
        collection_name=_COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    ).points

    context = []
    for point in results:
        payload = point.payload
        context.append({
            "policy_id": payload["policy_id"],
            "version": payload["version"],
            "title": payload["title"],
            "section": payload["section"],
            "content": payload["content"],
            "source_file": payload["source_file"],
            "score": point.score,  # cosine similarity, higher = more relevant (unlike Chroma's distance)
        })

    return context


if __name__ == "__main__":
    from chunker import chunk_all_policies
    from embedder import embed_chunks

    chunks = chunk_all_policies()
    embedded = embed_chunks(chunks)
    index_chunks(embedded)
    print(f"Indexed {len(embedded)} chunks into Qdrant at {_QDRANT_HOST}:{_QDRANT_PORT}")

    sample = retrieve_policy_context("dormant account handling requirements", top_k=2)
    for r in sample:
        print(f"  [{r['policy_id']}] {r['section']} (score={r['score']:.3f})")