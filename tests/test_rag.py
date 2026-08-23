"""
Tests for the RAG layer (rag/chunker.py, rag/embedder.py, rag/vector_store.py).

Split into three tiers:
  1. Chunking tests — no external dependencies, always run.
  2. Cross-validation against controls.json — no external dependencies,
     always run. This is the test that would have caught the old
     DORMANT_001 policy_id bug automatically (see §5 Step 1 of the
     pipeline status report).
  3. Embedding + retrieval integration tests — require a live Qdrant
     server on localhost:6333 and network access to download the
     embedding model on first run. These are skipped automatically
     if Qdrant isn't reachable, so `pytest` stays green in CI/offline
     environments without silently hiding a real regression.
"""

import json
import socket
from pathlib import Path

import pytest

from RAG.chunker import chunk_all_policies, parse_policy_file


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CONTROLS_PATH = DATA_DIR  / "controls.json"


def _qdrant_is_reachable(host: str = "localhost", port: int = 6333) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------
# Tier 1 — Chunking
# ---------------------------------------------------------------------

def test_chunk_all_policies_parses_every_policy_file():
    chunks = chunk_all_policies()

    policy_files = sorted(DATA_DIR.glob("0*_*.md"))
    assert len(policy_files) == 6, (
        "Expected 6 policy markdown files in data/ — if this changed, "
        "the test corpus assumptions below need updating too."
    )
    assert len(chunks) > 0

    # Every policy file must be represented by at least one chunk.
    seen_source_files = {c["source_file"] for c in chunks}
    expected_source_files = {f.name for f in policy_files}
    assert seen_source_files == expected_source_files


def test_each_chunk_has_required_fields():
    chunks = chunk_all_policies()
    required_keys = {"policy_id", "version", "title", "section", "content", "source_file"}

    for chunk in chunks:
        assert required_keys.issubset(chunk.keys())
        assert chunk["policy_id"], "policy_id must not be empty"
        assert chunk["section"], "section must not be empty"
        assert chunk["content"].strip(), "content must not be empty/whitespace-only"


def test_no_duplicate_policy_id_section_pairs():
    """Each (policy_id, section) pair must be unique — this is the
    identity vector_store.py uses for Qdrant point IDs, so a
    duplicate would silently overwrite a chunk during indexing."""
    chunks = chunk_all_policies()
    pairs = [(c["policy_id"], c["section"]) for c in chunks]
    assert len(pairs) == len(set(pairs))


def test_parse_policy_file_rejects_missing_policy_id(tmp_path):
    malformed = tmp_path / "01_bad_policy.md"
    malformed.write_text(
        "# Some Policy\nVersion: 1.0\n\n## Requirements\nDo the thing.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Policy ID"):
        parse_policy_file(malformed)


def test_parse_policy_file_rejects_missing_sections(tmp_path):
    malformed = tmp_path / "01_bad_policy.md"
    malformed.write_text(
        "# Some Policy\nPolicy ID: TEST-POLICY-001\nVersion: 1.0\n\nJust prose, no ## headers.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no ## sections"):
        parse_policy_file(malformed)


def test_parse_policy_file_rejects_missing_title(tmp_path):
    malformed = tmp_path / "01_bad_policy.md"
    malformed.write_text(
        "Policy ID: TEST-POLICY-001\nVersion: 1.0\n\n## Requirements\nDo the thing.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="H1 title"):
        parse_policy_file(malformed)


# ---------------------------------------------------------------------
# Tier 2 — Cross-validation against controls.json
#
# This is the highest-value test in this file: it guarantees every
# policy_id + section referenced by a control actually resolves to
# real chunked content. This is the exact check that would have
# caught the old DORMANT_001 -> "DORMANT_001" (should be
# "DORMANT-POLICY-001") bug automatically instead of by manual review.
# ---------------------------------------------------------------------

def test_every_control_policy_reference_resolves_in_rag_chunks():
    chunks = chunk_all_policies()
    chunk_index = {(c["policy_id"], c["section"]) for c in chunks}

    controls = json.loads(CONTROLS_PATH.read_text(encoding="utf-8"))
    assert controls, "controls.json is empty or missing"

    unresolved = []
    for control in controls:
        for ref in control.get("policy_references", []):
            key = (ref["policy_id"], ref["section"])
            if key not in chunk_index:
                unresolved.append((control["control_id"], key))

    assert not unresolved, (
        f"controls.json references policy sections that don't exist in the "
        f"chunked policy content: {unresolved}"
    )


def test_every_control_has_at_least_one_policy_reference():
    controls = json.loads(CONTROLS_PATH.read_text(encoding="utf-8"))
    missing = [c["control_id"] for c in controls if not c.get("policy_references")]
    assert not missing, f"Controls with no policy_references: {missing}"


# ---------------------------------------------------------------------
# Tier 3 — Embedding + retrieval integration (requires Qdrant + network)
# ---------------------------------------------------------------------

requires_qdrant = pytest.mark.skipif(
    not _qdrant_is_reachable(),
    reason="Qdrant is not running on localhost:6333 — start it with "
    "`docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant` to run this test.",
)


@requires_qdrant
def test_embed_chunks_adds_embedding_vector_of_expected_dim():
    from RAG.embedder import embed_chunks

    chunks = chunk_all_policies()[:2]  # keep it fast — don't embed all 15 for a unit test
    embedded = embed_chunks(chunks)

    for chunk in embedded:
        assert "embedding" in chunk
        assert len(chunk["embedding"]) == 384
        assert all(isinstance(x, float) for x in chunk["embedding"])


@requires_qdrant
def test_index_and_retrieve_roundtrip():
    from RAG.embedder import embed_chunks
    from RAG.vector_store import index_chunks, retrieve_policy_context

    chunks = chunk_all_policies()
    embedded = embed_chunks(chunks)
    index_chunks(embedded)

    results = retrieve_policy_context("dormant account handling requirements", top_k=3)

    assert len(results) == 3
    assert results[0]["policy_id"] == "DORMANT-POLICY-001", (
        "Top retrieval result for a dormant-accounts query should be the "
        "dormant accounts policy — got a different policy instead, which "
        "suggests the embedding/index isn't discriminating correctly."
    )
    for r in results:
        required_keys = {"policy_id", "version", "title", "section", "content", "source_file", "score"}
        assert required_keys.issubset(r.keys())


@requires_qdrant
def test_retrieved_policy_ids_all_exist_in_controls_json():
    """Sanity check: a few representative queries should only ever
    surface policy_ids that controls.json actually knows about."""
    from RAG.vector_store import retrieve_policy_context

    controls = json.loads(CONTROLS_PATH.read_text(encoding="utf-8"))
    known_policy_ids = {
        ref["policy_id"]
        for control in controls
        for ref in control.get("policy_references", [])
    }

    queries = [
        "customer screening evidence requirements",
        "risk exception approval reference",
        "Arabic name normalization rules",
        "source to report reconciliation fields",
    ]

    for query in queries:
        results = retrieve_policy_context(query, top_k=1)
        assert results[0]["policy_id"] in known_policy_ids