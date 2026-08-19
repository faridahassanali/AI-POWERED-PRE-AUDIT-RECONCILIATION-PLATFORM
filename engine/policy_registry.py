from pathlib import Path

from engine.policy_loader import (
    load_policies,
    validate_policy,
)


class PolicyRegistryError(ValueError):
    """Raised when the policy registry receives an invalid policy."""


class PolicyRegistry:
    """
    Registry of validated policies indexed by policy_id.
    """

    def __init__(self, policies=None):
        self._policies = {}

        if policies:
            for policy in policies:
                self.register(policy)

    def register(self, policy: dict) -> None:
        """
        Validate and register one policy.

        A policy must contain:
            - policy_id
            - version
            - title
            - sections

        Duplicate policy IDs are rejected.
        """

        if not isinstance(policy, dict):
            raise PolicyRegistryError(
                "Policy must be a dictionary."
            )

        # Important:
        # The Registry validates independently of the Loader.
        # This prevents callers from registering incomplete
        # policies directly.
        try:
            validate_policy(policy)
        except Exception as exc:
            raise PolicyRegistryError(
                f"Invalid policy: {exc}"
            ) from exc

        policy_id = policy["policy_id"]

        if policy_id in self._policies:
            raise PolicyRegistryError(
                f"Duplicate policy_id: {policy_id}"
            )

        self._policies[policy_id] = policy

    def get(self, policy_id: str) -> dict:
        """
        Return a policy by policy_id.

        Raises KeyError if the policy does not exist.
        """

        try:
            return self._policies[policy_id]
        except KeyError:
            raise KeyError(
                f"Policy not found: {policy_id}"
            ) from None

    def resolve(self, policy_id: str) -> dict | None:
        """
        Return a policy by policy_id.

        Returns None if the policy does not exist.
        """

        return self._policies.get(policy_id)

    def contains(self, policy_id: str) -> bool:
        """Return True if a policy is registered."""

        return policy_id in self._policies

    def all(self) -> list[dict]:
        """Return all registered policies."""

        return list(self._policies.values())

    def ids(self) -> set[str]:
        """Return all registered policy IDs."""

        return set(self._policies.keys())

    def __len__(self) -> int:
        return len(self._policies)

    def __contains__(self, policy_id: str) -> bool:
        return self.contains(policy_id)


def load_policy_registry(data_dir: Path) -> PolicyRegistry:
    """
    Load all policies from data_dir and build a registry.

    The PolicyRegistry will validate every policy again when
    registering it.
    """

    policies = load_policies(data_dir)

    return PolicyRegistry(policies)