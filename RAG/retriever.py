"""
RAG Policy Retriever
====================

Retrieves policy context for audit findings.

Retrieval strategy
------------------

1. If a finding contains policy_references:
   - The referenced policy is authoritative.
   - If policy_id + version + section are supplied, resolve exactly.
   - If only policy_id is supplied, resolve only that policy.
   - Never switch to a different policy.

2. If a finding does not contain the policy_references field:
   - Fall back to lexical retrieval.

3. If policy_references is explicitly present but empty:
   - Raise an error because the finding is expected to be grounded
     to an authoritative policy reference.

This keeps the RAG layer grounded while remaining compatible with
the existing project tests and finding format.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


# =====================================================================
# STOP WORDS
# =====================================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


# =====================================================================
# GENERIC HELPERS
# =====================================================================

def _get_field(
    obj: Any,
    field: str,
    default: Any = None,
) -> Any:
    """
    Read a field from either a dictionary or an object.
    """

    if isinstance(obj, dict):
        return obj.get(field, default)

    return getattr(obj, field, default)


def _as_list(value: Any) -> list[Any]:
    """
    Convert common collection formats into a list.
    """

    if value is None:
        return []

    if isinstance(value, dict):
        return list(value.values())

    if isinstance(value, (list, tuple, set)):
        return list(value)

    return [value]


def _tokenize(value: Any) -> set[str]:
    """
    Convert text into normalized searchable tokens.
    """

    if value is None:
        return set()

    tokens = re.findall(
        r"[a-z0-9_]+",
        str(value).lower(),
    )

    return {
        token
        for token in tokens
        if token not in STOP_WORDS
    }


# =====================================================================
# POLICY REGISTRY HELPERS
# =====================================================================

def _get_registry_policies(
    registry: Any,
) -> list[Any]:
    """
    Extract policies from the Policy Registry.

    Supported representations:

    - dictionary
    - registry.policies
    - registry._policies
    - registry.all()
    """

    if isinstance(registry, dict):
        return list(registry.values())

    policies = _get_field(
        registry,
        "policies",
    )

    if policies is not None:
        return _as_list(policies)

    policies = _get_field(
        registry,
        "_policies",
    )

    if policies is not None:
        return _as_list(policies)

    all_method = getattr(
        registry,
        "all",
        None,
    )

    if callable(all_method):
        return _as_list(
            all_method()
        )

    raise TypeError(
        "Could not find policies in the Policy Registry."
    )


def _get_policy_sections(
    policy: Any,
) -> list[Any]:
    """
    Get sections belonging to one policy.
    """

    return _as_list(
        _get_field(
            policy,
            "sections",
            [],
        )
    )


def _get_section_name(
    section: Any,
) -> str:
    """
    Get a section name.
    """

    name = _get_field(
        section,
        "section",
    )

    if name is None:
        name = _get_field(
            section,
            "name",
            "",
        )

    return str(name)


def _get_section_content(
    section: Any,
) -> str:
    """
    Get section content.
    """

    return str(
        _get_field(
            section,
            "content",
            "",
        )
    )


# =====================================================================
# REGISTRY -> SEARCHABLE CHUNKS
# =====================================================================

def build_policy_chunks(
    registry: Any,
) -> list[dict[str, Any]]:
    """
    Convert every Policy Registry section into a searchable chunk.

    The Registry remains the authoritative source.
    """

    chunks: list[dict[str, Any]] = []

    policies = _get_registry_policies(
        registry
    )

    for policy in policies:

        policy_id = _get_field(
            policy,
            "policy_id",
        )

        version = _get_field(
            policy,
            "version",
        )

        title = _get_field(
            policy,
            "title",
        )

        if not policy_id:
            raise ValueError(
                "Policy is missing policy_id."
            )

        if not version:
            raise ValueError(
                f"Policy '{policy_id}' is missing version."
            )

        if not title:
            raise ValueError(
                f"Policy '{policy_id}' is missing title."
            )

        sections = _get_policy_sections(
            policy
        )

        if not sections:
            raise ValueError(
                f"Policy '{policy_id}' has no sections."
            )

        for section in sections:

            section_name = _get_section_name(
                section
            )

            content = _get_section_content(
                section
            )

            if not section_name:
                raise ValueError(
                    f"Policy '{policy_id}' "
                    "has a section without a name."
                )

            if not content.strip():
                continue

            chunks.append(
                {
                    "policy_id": str(
                        policy_id
                    ),
                    "version": str(
                        version
                    ),
                    "title": str(
                        title
                    ),
                    "section": section_name,
                    "content": content,
                }
            )

    return chunks


# =====================================================================
# LEXICAL SCORING
# =====================================================================

def _score_chunk(
    query_tokens: set[str],
    chunk: dict[str, Any],
) -> float:
    """
    Calculate deterministic lexical relevance.

    Weighting:

    title   = 3
    section = 2
    content = 1
    """

    if not query_tokens:
        return 0.0

    title_tokens = _tokenize(
        chunk["title"]
    )

    section_tokens = _tokenize(
        chunk["section"]
    )

    content_tokens = _tokenize(
        chunk["content"]
    )

    title_matches = (
        query_tokens & title_tokens
    )

    section_matches = (
        query_tokens & section_tokens
    )

    content_matches = (
        query_tokens & content_tokens
    )

    return (
        len(title_matches) * 3.0
        + len(section_matches) * 2.0
        + len(content_matches)
    )


# =====================================================================
# GENERAL POLICY RETRIEVAL
# =====================================================================

def retrieve_policy(
    query: str,
    registry: Any,
    top_k: int = 3,
    policy_ids: Iterable[str] | None = None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """
    General lexical policy retrieval.

    This function can optionally restrict retrieval to:

    - specific policy IDs
    - a specific section
    """

    if top_k <= 0:
        return []

    if query is None or not str(query).strip():
        return []

    chunks = build_policy_chunks(
        registry
    )

    allowed_policy_ids = None

    if policy_ids is not None:

        allowed_policy_ids = {
            str(policy_id)
            for policy_id in policy_ids
        }

        known_policy_ids = {
            chunk["policy_id"]
            for chunk in chunks
        }

        unknown_policy_ids = (
            allowed_policy_ids
            - known_policy_ids
        )

        if unknown_policy_ids:
            raise ValueError(
                "Unknown policy ID(s): "
                + ", ".join(
                    sorted(
                        unknown_policy_ids
                    )
                )
            )

    normalized_section = None

    if section is not None:

        normalized_section = (
            str(section)
            .strip()
            .lower()
        )

    query_tokens = _tokenize(
        query
    )

    if not query_tokens:
        return []

    results = []

    for chunk in chunks:

        if (
            allowed_policy_ids is not None
            and chunk["policy_id"]
            not in allowed_policy_ids
        ):
            continue

        if (
            normalized_section is not None
            and chunk["section"]
            .strip()
            .lower()
            != normalized_section
        ):
            continue

        score = _score_chunk(
            query_tokens,
            chunk
        )

        if score <= 0:
            continue

        result = dict(chunk)

        result["relevance_score"] = score

        results.append(result)

    results.sort(
        key=lambda item: (
            -item["relevance_score"],
            item["policy_id"],
            item["section"],
        )
    )

    return results[:top_k]


# =====================================================================
# FINDING POLICY REFERENCE EXTRACTION
# =====================================================================

def _extract_policy_references(
    finding: Any,
) -> list[dict[str, str]]:
    """
    Extract policy references from a finding.

    Supported formats:

    Format 1 - full reference:

        {
            "policy_id": "SCREENING-POLICY-001",
            "version": "1.0",
            "section": "Requirements"
        }

    Format 2 - legacy project format:

        "SCREENING-POLICY-001"
    """

    references = _get_field(
        finding,
        "policy_references",
        None,
    )

    # Field completely absent.
    if references is None:
        return []

    if isinstance(references, dict):
        references = [references]

    if isinstance(references, str):
        references = [references]

    result: list[dict[str, str]] = []

    for reference in references:

        # -------------------------------------------------------------
        # Legacy policy-id-only reference
        # -------------------------------------------------------------

        if isinstance(reference, str):

            policy_id = reference.strip()

            if not policy_id:
                continue

            result.append(
                {
                    "policy_id": policy_id,
                    "version": "",
                    "section": "",
                }
            )

            continue

        # -------------------------------------------------------------
        # Full reference object
        # -------------------------------------------------------------

        policy_id = _get_field(
            reference,
            "policy_id",
        )

        version = _get_field(
            reference,
            "version",
        )

        section = _get_field(
            reference,
            "section",
        )

        if not policy_id:
            raise ValueError(
                "Policy reference is missing policy_id."
            )

        result.append(
            {
                "policy_id": str(
                    policy_id
                ),
                "version": (
                    str(version)
                    if version is not None
                    else ""
                ),
                "section": (
                    str(section)
                    if section is not None
                    else ""
                ),
            }
        )

    return result


# =====================================================================
# EXACT POLICY RESOLUTION
# =====================================================================

def resolve_policy_references(
    finding: Any,
    registry: Any,
) -> list[dict[str, Any]]:
    """
    Resolve policy references directly against the Policy Registry.

    This is the most important grounding function.

    If the finding says:

        SCREENING-POLICY-001

    the retriever can ONLY return chunks belonging to:

        SCREENING-POLICY-001

    It cannot replace it with a similar policy.

    If version and section are also provided, those are matched exactly.
    """

    references = _extract_policy_references(
        finding
    )

    if not references:
        raise ValueError(
            "Finding has no policy_references. "
            "Referenced policy retrieval requires "
            "at least one policy reference."
        )

    chunks = build_policy_chunks(
        registry
    )

    resolved: list[dict[str, Any]] = []

    for reference in references:

        policy_id = reference[
            "policy_id"
        ]

        version = reference[
            "version"
        ]

        section = reference[
            "section"
        ]

        # -------------------------------------------------------------
        # FULL EXACT REFERENCE
        # -------------------------------------------------------------

        if version and section:

            matches = [
                chunk
                for chunk in chunks
                if (
                    chunk["policy_id"]
                    == policy_id
                    and chunk["version"]
                    == version
                    and chunk["section"]
                    .strip()
                    .lower()
                    == section
                    .strip()
                    .lower()
                )
            ]

        # -------------------------------------------------------------
        # SECTION-ONLY REFERENCE (version not supplied)
        #
        # Still filter by the section that WAS given -- a partial
        # reference should narrow down, not fall all the way back to
        # "every section of this policy".
        # -------------------------------------------------------------

        elif section:

            matches = [
                chunk
                for chunk in chunks
                if (
                    chunk["policy_id"]
                    == policy_id
                    and chunk["section"]
                    .strip()
                    .lower()
                    == section
                    .strip()
                    .lower()
                )
            ]

        # -------------------------------------------------------------
        # VERSION-ONLY REFERENCE (section not supplied)
        # -------------------------------------------------------------

        elif version:

            matches = [
                chunk
                for chunk in chunks
                if (
                    chunk["policy_id"]
                    == policy_id
                    and chunk["version"]
                    == version
                )
            ]

        # -------------------------------------------------------------
        # POLICY-ID-ONLY REFERENCE
        #
        # IMPORTANT:
        #
        # We search ONLY this policy.
        # We never search the complete registry semantically.
        # -------------------------------------------------------------

        else:

            matches = [
                chunk
                for chunk in chunks
                if (
                    chunk["policy_id"]
                    == policy_id
                )
            ]

        # -------------------------------------------------------------
        # REFERENCE NOT FOUND
        # -------------------------------------------------------------

        if not matches:

            if version and section:

                raise ValueError(
                    "Policy reference could not be "
                    "resolved exactly in the Policy Registry: "
                    f"{policy_id} "
                    f"version {version} "
                    f"section '{section}'."
                )

            if section:

                raise ValueError(
                    "Policy reference could not be "
                    "resolved in the Policy Registry: "
                    f"{policy_id} "
                    f"section '{section}' "
                    "(no version specified)."
                )

            if version:

                raise ValueError(
                    "Policy reference could not be "
                    "resolved in the Policy Registry: "
                    f"{policy_id} "
                    f"version {version} "
                    "(no section specified)."
                )

            raise ValueError(
                "Policy ID could not be resolved "
                "in the Policy Registry: "
                f"{policy_id}"
            )

        # -------------------------------------------------------------
        # EXACT REFERENCE MUST BE UNIQUE
        # -------------------------------------------------------------

        if (
            version
            and section
            and len(matches) > 1
        ):

            raise ValueError(
                "Policy Registry contains duplicate "
                "policy sections for "
                f"{policy_id} "
                f"version {version} "
                f"section '{section}'."
            )

        # -------------------------------------------------------------
        # RETURN RESOLVED CHUNKS
        # -------------------------------------------------------------

        for match in matches:

            result = dict(match)

            # Exact resolution is not semantic ranking.
            result["relevance_score"] = 1.0

            resolved.append(
                result
            )

    return resolved


# =====================================================================
# BUILD FINDING QUERY
# =====================================================================

def _build_finding_query(
    finding: Any,
) -> str:
    """
    Build a lexical search query from finding information.
    """

    fields = [
        "control_id",
        "control_name",
        "description",
        "expected",
        "expected_value",
        "actual",
        "actual_value",
        "evidence",
        "reason",
    ]

    values: list[str] = []

    for field in fields:

        value = _get_field(
            finding,
            field,
            "",
        )

        if value is None:
            continue

        values.append(
            str(value)
        )

    return " ".join(values)


# =====================================================================
# MAIN FINDING RETRIEVAL
# =====================================================================

def retrieve_for_finding(
    finding: Any,
    registry: Any,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve policy context for an audit finding.

    IMPORTANT BEHAVIOR
    ------------------

    A finding with an explicit policy_references field is expected
    to be grounded to those references.

    Example:

        {
            "policy_references": [
                "SCREENING-POLICY-001"
            ]
        }

    returns ONLY SCREENING-POLICY-001.

    It cannot switch to another policy.

    If the field is completely absent, lexical fallback retrieval
    remains available for generic/unlinked findings.
    """

    if finding is None:
        raise ValueError(
            "finding cannot be None."
        )

    # Check whether the field actually exists.
    #
    # This is intentionally different from simply checking the
    # extracted list.
    #
    # Why?
    #
    # policy_references: []
    #
    # means the finding explicitly has no authoritative reference.
    # The tests/project expect this to be rejected.
    has_reference_field = (
        isinstance(finding, dict)
        and "policy_references" in finding
    )

    if not isinstance(finding, dict):
        has_reference_field = hasattr(
            finding,
            "policy_references",
        )

    references = _extract_policy_references(
        finding
    )

    # -------------------------------------------------------------
    # PRIMARY PATH
    #
    # Explicit policy references.
    # -------------------------------------------------------------

    if references:

        return resolve_policy_references(
            finding=finding,
            registry=registry,
        )

    # -------------------------------------------------------------
    # EXPLICIT BUT EMPTY REFERENCES
    #
    # Do NOT silently perform semantic fallback.
    #
    # This prevents a finding that was supposed to be grounded
    # from accidentally receiving a similar but incorrect policy.
    # -------------------------------------------------------------

    if has_reference_field:

        raise ValueError(
            "Finding has no policy_references. "
            "A finding must contain at least one "
            "authoritative policy reference."
        )

    # -------------------------------------------------------------
    # FALLBACK PATH
    #
    # Only used when the finding does not contain the field at all.
    #
    # Tries semantic (embeddings/Qdrant) retrieval first, since it
    # generally beats plain lexical token-overlap for this kind of
    # unlinked/generic query -- especially for multilingual content
    # (e.g. Arabic policy text). Falls back to lexical retrieval if
    # Qdrant/embeddings aren't available or return nothing, so this
    # never hard-fails just because a Qdrant server isn't running.
    #
    # This is intentionally scoped to the fallback branch only. The
    # primary path above (explicit policy_references) NEVER goes
    # through semantic search -- an audit finding must stay grounded
    # to the exact policy it names, not a similar one a vector search
    # happened to surface.
    # -------------------------------------------------------------

    query = _build_finding_query(
        finding
    )

    if not query.strip():
        return []

    semantic_results = _try_semantic_fallback(
        query=query,
        top_k=top_k,
    )

    if semantic_results:
        return semantic_results

    return retrieve_policy(
        query=query,
        registry=registry,
        top_k=top_k,
    )


# =====================================================================
# SEMANTIC (EMBEDDINGS/QDRANT) FALLBACK
# =====================================================================

def _try_semantic_fallback(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Best-effort semantic retrieval via RAG.vector_store (Qdrant +
    sentence-transformer embeddings from RAG/embedder.py, RAG/chunker.py).

    Used ONLY by the lexical-fallback branch of retrieve_for_finding()
    -- never for findings with explicit policy_references, which are
    always resolved exactly against the registry instead.

    Returns [] rather than raising if Qdrant isn't running, the
    collection hasn't been indexed yet, or the embedding model can't
    be loaded -- so callers without that infra transparently get the
    lexical fallback instead of a crash. This is deliberate: this
    function must never be the reason retrieve_for_finding() fails.
    """

    try:
        from RAG.vector_store import retrieve_policy_context
    except ImportError:
        return []

    try:
        results = retrieve_policy_context(
            query,
            top_k=top_k,
        )
    except Exception:
        # Qdrant unreachable, collection not indexed, embedding model
        # missing, etc. -- any of these should fall back silently.
        return []

    # Normalize to the same shape retrieve_policy() returns
    # (relevance_score), so callers don't need to branch on which
    # fallback actually served the request.
    normalized: list[dict[str, Any]] = []

    for result in results:
        normalized_result = dict(result)
        normalized_result["relevance_score"] = result.get("score", 0.0)
        normalized.append(normalized_result)

    return normalized


# =====================================================================
# BACKWARD-COMPATIBLE ALIASES
# =====================================================================

get_policy_context = retrieve_policy

get_verified_policy_context = retrieve_for_finding