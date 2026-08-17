import json
import unittest
from pathlib import Path

from engine.policy_loader import (
    PolicyLoadError,
    load_policies,
    parse_policy,
)

from engine.policy_registry import (
    PolicyRegistry,
    PolicyRegistryError,
    load_policy_registry,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


EXPECTED_POLICY_IDS = {
    "SCREENING-POLICY-001",
    "RISK-POLICY-001",
    "DATA-POLICY-001",
    "DORMANT-POLICY-001",
    "WALLET-POLICY-001",
    "RECON-POLICY-001",
}


class TestPolicyLoader(unittest.TestCase):

    def test_all_six_policies_load(self):
        policies = load_policies(DATA_DIR)

        self.assertEqual(len(policies), 6)

        policy_ids = {
            policy["policy_id"]
            for policy in policies
        }

        self.assertEqual(policy_ids, EXPECTED_POLICY_IDS)

    def test_policy_metadata_is_loaded(self):
        policies = load_policies(DATA_DIR)

        for policy in policies:
            self.assertTrue(policy["policy_id"])
            self.assertTrue(policy["version"])
            self.assertTrue(policy["title"])
            self.assertTrue(policy["sections"])

    def test_sections_are_structured(self):
        policies = load_policies(DATA_DIR)

        for policy in policies:
            for section in policy["sections"]:
                self.assertEqual(
                    set(section.keys()),
                    {"section", "content"},
                )

                self.assertTrue(section["section"])
                self.assertIsInstance(
                    section["content"],
                    str,
                )

    def test_policy_content_is_preserved_verbatim(self):
        path = DATA_DIR / "01_customer_screening_policy.md"

        original = path.read_text(encoding="utf-8")

        policy = parse_policy(original)

        requirements = next(
            section
            for section in policy["sections"]
            if section["section"] == "Requirements"
        )

        start_marker = "## Requirements\n"
        end_marker = "## Audit Evidence"

        start = original.index(start_marker) + len(start_marker)
        end = original.index(end_marker)

        expected = original[start:end]

        self.assertEqual(
            requirements["content"],
            expected,
        )

    def test_malformed_policy_is_rejected(self):
        malformed = """\
# Broken Policy

This policy has no policy ID or version.

## Requirements

Something.
"""

        with self.assertRaises(PolicyLoadError):
            parse_policy(malformed)

    def test_duplicate_policy_ids_are_rejected(self):
        policy = {
            "policy_id": "TEST-POLICY",
            "version": "1.0",
            "title": "Test",
            "sections": [
                {
                    "section": "Requirements",
                    "content": "Test",
                }
            ],
        }

        from engine.policy_registry import PolicyRegistry

        registry = PolicyRegistry()

        registry.register(policy)

        with self.assertRaises(PolicyRegistryError):
            registry.register(policy)

    def test_registry_resolves_all_policies(self):
        registry = load_policy_registry(DATA_DIR)

        self.assertEqual(len(registry), 6)

        for policy_id in EXPECTED_POLICY_IDS:
            self.assertTrue(registry.contains(policy_id))
            self.assertIsNotNone(
                registry.resolve(policy_id)
            )

    def test_every_controls_json_policy_resolves(self):
        controls_path = DATA_DIR / "controls.json"

        controls = json.loads(
            controls_path.read_text(encoding="utf-8")
        )

        referenced_policy_ids = self._extract_policy_ids(
            controls
        )

        registry = load_policy_registry(DATA_DIR)

        missing = {
            policy_id
            for policy_id in referenced_policy_ids
            if not registry.contains(policy_id)
        }

        self.assertEqual(missing, set())

    def _extract_policy_ids(self, value):
        found = set()

        if isinstance(value, dict):
            if "policy_id" in value:
                found.add(value["policy_id"])

            for child in value.values():
                found.update(
                    self._extract_policy_ids(child)
                )

        elif isinstance(value, list):
            for child in value:
                found.update(
                    self._extract_policy_ids(child)
                )

        return found


if __name__ == "__main__":
    unittest.main()