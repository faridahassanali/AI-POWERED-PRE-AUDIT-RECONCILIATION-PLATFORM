from pathlib import Path

from engine.policy_loader import load_policies


class PolicyRegistryError(ValueError):
    """Raised when the policy registry is invalid."""


class PolicyRegistry:
    """
    Registry of policies indexed by policy_id.
    """

    def __init__(self, policies=None):
        self._policies = {}

        if policies:
            for policy in policies:
                self.register(policy)

    def register(self, policy: dict) -> None:
        """Register one policy."""

        policy_id = policy.get("policy_id")

        if not policy_id:
            raise PolicyRegistryError(
                "Cannot register a policy without policy_id."
            )

        if policy_id in self._policies:
            raise PolicyRegistryError(
                f"Duplicate policy_id: {policy_id}"
            )

        self._policies[policy_id] = policy

    def get(self, policy_id: str) -> dict:
        """
        Return a policy by ID.

        Raises KeyError if it doesn't exist.
        """

        try:
            return self._policies[policy_id]
        except KeyError:
            raise KeyError(
                f"Policy not found: {policy_id}"
            ) from None

    def resolve(self, policy_id: str) -> dict | None:
        """Return a policy or None if it doesn't exist."""

        return self._policies.get(policy_id)

    def contains(self, policy_id: str) -> bool:
        """Check whether a policy exists."""

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
    """Load all policies and build a registry."""

    policies = load_policies(data_dir)

    return PolicyRegistry(policies)