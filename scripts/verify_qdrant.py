"""
Qdrant Preflight Verification.

Standalone check run BEFORE relying on RAG retrieval, to catch a
missing/unreachable Qdrant server or an un-indexed collection loudly
instead of failing mysteriously later (e.g. retrieve_for_finding()
silently falling back to lexical search with no explanation of why
semantic retrieval never kicked in).

Mirrors scripts/verify_database.py's structure and philosophy:
one script, several named checks, a clear PASS/FAIL report, exit
code 0 only if everything passes.

Checks, in order:
    1. Connection        -- can we reach the Qdrant server at all?
    2. Collection exists  -- does "policy_chunks" exist?
    3. Vector config       -- is the configured vector size/distance
                              what RAG/vector_store.py expects (384,
                              cosine)?
    4. Indexed points       -- does the collection actually contain
                              points (i.e. has index_chunks() been
                              run), not just an empty shell?
    5. Point count sanity    -- does the point count roughly match
                              the number of chunks chunk_all_policies()
                              produces, so a partial/stale index is
                              visible rather than silently accepted?
    6. Embedding model loads   -- can the configured sentence-transformers
                              model actually be loaded (catches a
                              missing/corrupted local model cache
                              before the first real query hits it)?
    7. Round-trip retrieval     -- does a real query against the
                              dormant-accounts policy return the
                              dormant-accounts policy as the top hit?
                              (Same sanity check RAG/vector_store.py's
                              own __main__ block does manually.)

Usage
-----
    python scripts/verify_qdrant.py

    # or point at a non-default host/port:
    QDRANT_HOST=localhost QDRANT_PORT=6333 python scripts/verify_qdrant.py

Exit code is 0 only if every check passes. Non-zero otherwise, so
this is safe to wire into CI / a pre-deploy step alongside
verify_database.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# EXPECTED CONFIG
#
# Kept in sync with RAG/vector_store.py and RAG/embedder.py. If either
# changes (model, dimension, collection name), update this in the
# SAME PR -- this script is only useful if it actually reflects them.
# =====================================================================

EXPECTED_COLLECTION_NAME = "policy_chunks"
EXPECTED_VECTOR_SIZE = 384
EXPECTED_DISTANCE = "Cosine"
EXPECTED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))


# =====================================================================
# RESULT TRACKING (same shape as scripts/verify_database.py)
# =====================================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    details: list[str] = field(default_factory=list)


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, details: list[str] | None = None) -> None:
        self.results.append(CheckResult(name=name, passed=passed, details=details or []))

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def render(self) -> str:
        lines = []
        lines.append("=" * 64)
        lines.append("QDRANT PREFLIGHT VERIFICATION")
        lines.append(f"target: {QDRANT_HOST}:{QDRANT_PORT}")
        lines.append("=" * 64)

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append("")
            lines.append(f"[{status}] {result.name}")
            for detail in result.details:
                lines.append(f"    - {detail}")

        lines.append("")
        lines.append("=" * 64)
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        lines.append(f"RESULT: {passed}/{total} checks passed")
        lines.append("=" * 64)

        return "\n".join(lines)


# =====================================================================
# CHECKS
# =====================================================================

def check_connection(report: Report):
    """Returns the connected client, or None if unreachable."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        report.add(
            "Connection",
            False,
            ["qdrant-client is not installed. Run: pip install qdrant-client"],
        )
        return None

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        # get_collections() is the cheapest real round-trip -- forces
        # an actual request rather than just constructing the client.
        client.get_collections()
    except Exception as exc:
        report.add(
            "Connection",
            False,
            [
                f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}: {exc}",
                "Start it with: docker run -p 6333:6333 -p 6334:6334 "
                "-v $(pwd)/rag/.qdrant_storage:/qdrant/storage qdrant/qdrant",
            ],
        )
        return None

    report.add("Connection", True, [f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}."])
    return client


def check_collection_exists(report: Report, client) -> bool:
    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception as exc:
        report.add("Collection Exists", False, [f"Could not list collections: {exc}"])
        return False

    if EXPECTED_COLLECTION_NAME not in collections:
        report.add(
            "Collection Exists",
            False,
            [
                f"Collection '{EXPECTED_COLLECTION_NAME}' not found. "
                f"Existing collections: {collections or '(none)'}",
                "Run RAG/vector_store.py's __main__ block (or index_chunks()) "
                "to create and populate it.",
            ],
        )
        return False

    report.add("Collection Exists", True, [f"Collection '{EXPECTED_COLLECTION_NAME}' exists."])
    return True


def check_vector_config(report: Report, client) -> None:
    try:
        info = client.get_collection(EXPECTED_COLLECTION_NAME)
        vectors_config = info.config.params.vectors
        size = vectors_config.size
        distance = str(vectors_config.distance)
    except Exception as exc:
        report.add("Vector Config", False, [f"Could not read collection config: {exc}"])
        return

    problems = []

    if size != EXPECTED_VECTOR_SIZE:
        problems.append(
            f"Vector size is {size}, expected {EXPECTED_VECTOR_SIZE} "
            f"(the embedder model '{EXPECTED_MODEL_NAME}' output dimension). "
            "This usually means the collection was created with a different "
            "embedding model than the one currently configured."
        )

    if EXPECTED_DISTANCE.lower() not in distance.lower():
        problems.append(f"Distance metric is {distance}, expected {EXPECTED_DISTANCE}.")

    if problems:
        report.add("Vector Config", False, problems)
    else:
        report.add(
            "Vector Config",
            True,
            [f"size={size}, distance={distance} -- matches expected embedder output."],
        )


def check_indexed_points(report: Report, client) -> int:
    try:
        count_result = client.count(collection_name=EXPECTED_COLLECTION_NAME, exact=True)
        count = count_result.count
    except Exception as exc:
        report.add("Indexed Points", False, [f"Could not count points: {exc}"])
        return 0

    if count == 0:
        report.add(
            "Indexed Points",
            False,
            [
                "Collection exists but contains 0 points -- it has never been "
                "indexed. Run: python -m RAG.vector_store (or call "
                "index_chunks(embed_chunks(chunk_all_policies())) directly)."
            ],
        )
        return 0

    report.add("Indexed Points", True, [f"{count} point(s) indexed."])
    return count


def check_point_count_matches_corpus(report: Report, indexed_count: int) -> None:
    if indexed_count == 0:
        # Already reported as a failure by check_indexed_points; skip
        # the comparison so we don't double-report the same root cause.
        report.add("Point Count Sanity", False, ["Skipped -- no points indexed."])
        return

    try:
        from RAG.chunker import chunk_all_policies

        expected_chunks = len(chunk_all_policies())
    except Exception as exc:
        report.add(
            "Point Count Sanity",
            False,
            [f"Could not chunk the local policy corpus for comparison: {exc}"],
        )
        return

    if indexed_count != expected_chunks:
        report.add(
            "Point Count Sanity",
            False,
            [
                f"Qdrant has {indexed_count} point(s), but the local policy "
                f"corpus currently chunks into {expected_chunks}. The index is "
                "stale -- re-run indexing after any policy .md file change "
                "(index_chunks() upserts by policy_id+section, so a new "
                "section/policy file requires a fresh index run)."
            ],
        )
        return

    report.add(
        "Point Count Sanity",
        True,
        [f"Indexed point count ({indexed_count}) matches the local policy corpus."],
    )


def check_embedding_model_loads(report: Report) -> bool:
    try:
        from RAG.embedder import get_model

        model = get_model()
        dim = model.get_sentence_embedding_dimension()
    except Exception as exc:
        report.add(
            "Embedding Model Loads",
            False,
            [
                f"Could not load '{EXPECTED_MODEL_NAME}': {exc}",
                "First run downloads the model from HuggingFace -- make sure "
                "network access is available, or the model is already cached.",
            ],
        )
        return False

    if dim != EXPECTED_VECTOR_SIZE:
        report.add(
            "Embedding Model Loads",
            False,
            [f"Model loaded but outputs dimension {dim}, expected {EXPECTED_VECTOR_SIZE}."],
        )
        return False

    report.add("Embedding Model Loads", True, [f"'{EXPECTED_MODEL_NAME}' loaded, dim={dim}."])
    return True


def check_round_trip_retrieval(report: Report, model_ok: bool) -> None:
    if not model_ok:
        report.add("Round-Trip Retrieval", False, ["Skipped -- embedding model failed to load."])
        return

    try:
        from RAG.vector_store import retrieve_policy_context

        results = retrieve_policy_context("dormant account handling requirements", top_k=1)
    except Exception as exc:
        report.add("Round-Trip Retrieval", False, [f"Query failed: {exc}"])
        return

    if not results:
        report.add("Round-Trip Retrieval", False, ["Query returned zero results."])
        return

    top_policy_id = results[0].get("policy_id")

    if top_policy_id != "DORMANT-POLICY-001":
        report.add(
            "Round-Trip Retrieval",
            False,
            [
                "Sanity query for 'dormant account handling requirements' "
                f"returned '{top_policy_id}' as the top hit, expected "
                "'DORMANT-POLICY-001'. The index may be stale or corrupted."
            ],
        )
        return

    report.add(
        "Round-Trip Retrieval",
        True,
        [f"Sanity query correctly returned DORMANT-POLICY-001 (score={results[0].get('score'):.3f})."],
    )


# =====================================================================
# MAIN
# =====================================================================

def run_all_checks() -> Report:
    report = Report()

    client = check_connection(report)
    if client is None:
        return report

    collection_exists = check_collection_exists(report, client)
    if not collection_exists:
        return report

    check_vector_config(report, client)
    indexed_count = check_indexed_points(report, client)
    check_point_count_matches_corpus(report, indexed_count)
    model_ok = check_embedding_model_loads(report)
    check_round_trip_retrieval(report, model_ok)

    return report


def main() -> int:
    report = run_all_checks()
    print(report.render())
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
