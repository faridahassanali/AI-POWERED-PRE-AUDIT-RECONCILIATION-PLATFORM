"""
Layer 1 — Data Normalization.

Normalizes loaded source tables without changing their business meaning.

Normalization rules:
- Trim leading/trailing whitespace from string values.
- Normalize known categorical fields to canonical uppercase values.
- Normalize boolean-like fields to True/False strings.
- Keep free-text fields such as Arabic/English names unchanged except for trimming.
- Do not modify IDs or business values beyond representation cleanup.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------
# Canonical categorical fields
# ---------------------------------------------------------

CATEGORICAL_FIELDS = {
    "screening_status",
    "risk_level",
    "wallet_status",
    "account_status",
    "dormant_handling_status",
    "population_status",
}


# ---------------------------------------------------------
# Boolean-like fields
# ---------------------------------------------------------

BOOLEAN_FIELDS = {
    "screening_evidence_present",
    "risk_exception_approved",
}


TRUE_VALUES = {
    "true",
    "1",
    "yes",
    "y",
}

FALSE_VALUES = {
    "false",
    "0",
    "no",
    "n",
}


def _normalize_string(value: Any) -> Any:
    """
    Trim whitespace from string values.

    Non-string values are returned unchanged.
    """
    if isinstance(value, str):
        return value.strip()

    return value


def _normalize_categorical(value: Any) -> Any:
    """
    Normalize categorical values to uppercase.

    Example:
        " opened " -> "OPENED"
        "clear"    -> "CLEAR"
        " High "   -> "HIGH"
    """
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return ""

        return value.upper()

    return value


def _normalize_boolean(value: Any) -> Any:
    """
    Normalize boolean-like values into canonical string values.

    Examples:
        "true"  -> "True"
        "TRUE"  -> "True"
        "yes"   -> "True"
        "1"     -> "True"

        "false" -> "False"
        "no"    -> "False"
        "0"     -> "False"

    Empty values remain empty.
    Unknown values are preserved after trimming.
    """

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, str):
        normalized = value.strip().lower()

        if not normalized:
            return ""

        if normalized in TRUE_VALUES:
            return "True"

        if normalized in FALSE_VALUES:
            return "False"

        # Do not silently change unexpected values.
        return value.strip()

    return value


def normalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize one DataFrame.

    The original DataFrame is not modified.
    """

    normalized = df.copy()

    # -----------------------------------------------------
    # Step 1 — Trim all string cells
    # -----------------------------------------------------

    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            _normalize_string
        )

    # -----------------------------------------------------
    # Step 2 — Normalize categorical columns
    # -----------------------------------------------------

    for column in CATEGORICAL_FIELDS:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(
                _normalize_categorical
            )

    # -----------------------------------------------------
    # Step 3 — Normalize boolean-like columns
    # -----------------------------------------------------

    for column in BOOLEAN_FIELDS:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(
                _normalize_boolean
            )

    return normalized


def normalize_tables(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Normalize all loaded source tables.

    Returns a new dictionary and does not mutate the
    original loaded tables.
    """

    return {
        name: normalize_dataframe(df)
        for name, df in tables.items()
    }