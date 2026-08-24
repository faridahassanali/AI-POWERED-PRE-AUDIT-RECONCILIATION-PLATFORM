"""
Finding Schema Validator.

Validates standardized audit findings against
data/finding_schema.json.

This module is responsible only for structural/schema
validation. Business validation remains the responsibility
of the audit controls.
"""

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError


DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "finding_schema.json"
)


class FindingValidationError(Exception):
    """Raised when a finding does not comply with the finding schema."""


def load_finding_schema(
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """
    Load the finding JSON schema from disk.

    Args:
        schema_path: Path to finding_schema.json.

    Returns:
        The loaded JSON schema.

    Raises:
        FileNotFoundError:
            If the schema file does not exist.
        json.JSONDecodeError:
            If the schema file contains invalid JSON.
    """

    schema_path = Path(schema_path)

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Finding schema not found: {schema_path}"
        )

    with schema_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_validator(
    schema: dict[str, Any],
) -> Draft7Validator:
    """
    Create a JSON Schema validator for the finding schema.

    Args:
        schema: Loaded finding JSON schema.

    Returns:
        A configured Draft7Validator.

    Raises:
        SchemaError:
            If the provided schema itself is invalid.
    """

    try:
        return Draft7Validator(schema)
    except SchemaError as exc:
        raise SchemaError(
            f"Invalid finding schema: {exc.message}"
        ) from exc


def get_validation_errors(
    finding: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> list[str]:
    """
    Return all schema validation errors for a finding.

    Args:
        finding: Finding object to validate.
        schema: Optional loaded finding schema.
                If omitted, the default schema is loaded.

    Returns:
        A list of human-readable validation error messages.
        An empty list means the finding is valid.
    """

    if schema is None:
        schema = load_finding_schema()

    validator = create_validator(schema)

    errors = sorted(
        validator.iter_errors(finding),
        key=lambda error: list(error.path),
    )

    messages: list[str] = []

    for error in errors:
        if error.path:
            field_path = ".".join(str(part) for part in error.path)
            messages.append(
                f"{field_path}: {error.message}"
            )
        else:
            messages.append(error.message)

    return messages


def validate_finding(
    finding: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> bool:
    """
    Check whether a finding complies with the finding schema.

    Args:
        finding: Finding object to validate.
        schema: Optional loaded finding schema.
                If omitted, the default schema is loaded.

    Returns:
        True if the finding is valid, otherwise False.
    """

    return len(get_validation_errors(finding, schema)) == 0


def validate_finding_or_raise(
    finding: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> None:
    """
    Validate a finding and raise an exception if invalid.

    Args:
        finding: Finding object to validate.
        schema: Optional loaded finding schema.
                If omitted, the default schema is loaded.

    Raises:
        FindingValidationError:
            If the finding does not comply with the schema.
    """

    errors = get_validation_errors(finding, schema)

    if errors:
        error_message = "Finding validation failed:\n- " + "\n- ".join(
            errors
        )

        raise FindingValidationError(error_message)
def create_validator(
    schema: dict[str, Any],
) -> Draft7Validator:
    """
    Create a JSON Schema validator for the finding schema.

    Args:
        schema: Loaded finding JSON schema.

    Returns:
        A configured Draft7Validator.

    Raises:
        SchemaError:
            If the provided schema itself is invalid.
    """

    try:
        Draft7Validator.check_schema(schema)
        return Draft7Validator(schema)
    except SchemaError as exc:
        raise SchemaError(
            f"Invalid finding schema: {exc.message}"
        ) from exc    