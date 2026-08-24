"""
Tests for RAG.retriever.resolve_policy_references()'s handling of
PARTIAL policy references (only `version` or only `section` given,
not both).

FIXED BEHAVIOR (previously a review finding):

resolve_policy_references() used to only filter on version+section
when BOTH were supplied. If either was missing, it fell all the way
back to "policy-id-only" -- silently dropping whichever field WAS
present and returning every section of that policy. A reference with
`section="Requirements"` but no `version` would return every section
of the policy, not just Requirements.

Now, a partial reference filters on whichever field IS present:

    - section given, no version -> filtered to that section (across
      whatever versions match, though in practice the registry only
      ever holds one version per policy)
    - version given, no section -> filtered to that version, all of
      its sections
    - neither given -> policy-id-only (unchanged, pre-existing
      behavior)
    - both given -> exact match, unique result required (unchanged)

This is the same regression guard as before
(test_all_real_controls_supply_both_version_and_section) -- kept,
since every real policy_references entry in data/controls.json still
supplies both fields together, so this fix doesn't yet apply to any
live control. It's here so the correct behavior is defined and
covered before it's ever needed.
"""

from pathlib import Path

import pytest

from engine.policy_registry import load_policy_registry
from RAG.retriever import resolve_policy_references


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def real_registry():
    return load_policy_registry(DATA_DIR)


def test_section_without_version_filters_to_that_section_only(real_registry):
    finding = {
        "policy_references": [
            {
                "policy_id": "DORMANT-POLICY-001",
                "version": "",
                "section": "Requirements",
            }
        ]
    }

    results = resolve_policy_references(finding, real_registry)

    assert {r["section"] for r in results} == {"Requirements"}


def test_version_without_section_returns_all_sections_of_that_version(real_registry):
    """
    A version-only reference legitimately can't narrow down to one
    section (it wasn't given one) -- it returns every section of the
    matched version, which is correct, not a fallback quirk.
    """
    finding = {
        "policy_references": [
            {
                "policy_id": "DORMANT-POLICY-001",
                "version": "1.0",
                "section": "",
            }
        ]
    }

    results = resolve_policy_references(finding, real_registry)

    assert results
    assert all(r["policy_id"] == "DORMANT-POLICY-001" for r in results)
    assert all(r["version"] == "1.0" for r in results)


def test_partial_reference_never_crosses_into_a_different_policy(real_registry):
    finding = {
        "policy_references": [
            {
                "policy_id": "DORMANT-POLICY-001",
                "version": "",
                "section": "Requirements",
            }
        ]
    }

    results = resolve_policy_references(finding, real_registry)

    assert {r["policy_id"] for r in results} == {"DORMANT-POLICY-001"}


def test_full_reference_with_version_and_section_still_filters_to_exactly_one(
    real_registry,
):
    """Contrast case: both fields given -> exact, unique match (unchanged)."""
    finding = {
        "policy_references": [
            {
                "policy_id": "DORMANT-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ]
    }

    results = resolve_policy_references(finding, real_registry)

    assert len(results) == 1
    assert results[0]["section"] == "Requirements"


def test_section_only_reference_that_does_not_exist_raises(real_registry):
    """A section-only reference to a section that doesn't exist under
    the policy must still raise, not silently fall back to all sections."""
    finding = {
        "policy_references": [
            {
                "policy_id": "DORMANT-POLICY-001",
                "version": "",
                "section": "Does Not Exist",
            }
        ]
    }

    with pytest.raises(ValueError, match="Does Not Exist"):
        resolve_policy_references(finding, real_registry)


def test_version_only_reference_that_does_not_exist_raises(real_registry):
    finding = {
        "policy_references": [
            {
                "policy_id": "DORMANT-POLICY-001",
                "version": "99.9",
                "section": "",
            }
        ]
    }

    with pytest.raises(ValueError, match="99.9"):
        resolve_policy_references(finding, real_registry)


def test_all_real_controls_supply_both_version_and_section():
    """
    Not currently a live risk: every policy_references entry in
    data/controls.json supplies both version and section together.
    Kept as a guard so a future control that only supplies one of the
    two is at least flagged, even though the fixed behavior above now
    handles that case correctly rather than over-returning results.
    """
    import json

    with open(DATA_DIR / "controls.json") as f:
        controls = json.load(f)

    incomplete = []
    for control in controls:
        for ref in control.get("policy_references", []):
            if not ref.get("version") or not ref.get("section"):
                incomplete.append((control["control_id"], ref))

    assert not incomplete, (
        f"These controls have a policy_references entry missing "
        f"version and/or section: {incomplete}. This is now handled "
        f"correctly (filters on whatever field IS present) rather "
        f"than over-returning all sections -- confirm it's intended."
    )
