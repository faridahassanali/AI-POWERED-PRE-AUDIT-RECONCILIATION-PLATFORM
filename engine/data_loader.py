from pathlib import Path

import pandas as pd

from engine.normalization import normalize_tables


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(
    data_dir: Path = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Read every source CSV into a DataFrame."""

    files = {
        "customers": "customers.csv",
        "partner_wallet_requests": "partner_wallet_requests.csv",
        "approved_source_population": "approved_source_population.csv",
        "screening_results": "screening_results.csv",
        "wallet_initialization": "wallet_initialization.csv",
        "dormant_accounts_report": "dormant_accounts_report.csv",
        "final_wallet_audit_report": "final_wallet_audit_report.csv",
        "expected_findings": "expected_findings.csv",
    }

    tables: dict[str, pd.DataFrame] = {}
    missing = []

    for key, filename in files.items():

        path = data_dir / filename

        if not path.exists():
            missing.append(filename)
            continue

        tables[key] = (
            pd.read_csv(
                path,
                dtype=str,
            )
            .fillna("")
        )

    if missing:
        raise FileNotFoundError(
            f"These files are missing from the data/ folder: "
            f"{missing}. "
            f"Make sure you copied every V3 dataset file "
            f"into data/."
        )

    return tables


def load_and_normalize_data(
    data_dir: Path = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Load all source CSV files and normalize them.
    """

    tables = load_data(data_dir)

    return normalize_tables(tables)


def build_unified_customer_record(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Join approved_source_population + customers +
    screening_results into one row per customer_id.
    """

    population = tables["approved_source_population"]

    customers = tables["customers"]

    screening = tables["screening_results"][
        [
            "customer_id",
            "screening_evidence_present",
            "screening_reference",
        ]
    ]

    population_flag = population[
        [
            "customer_id",
            "population_status",
        ]
    ]

    unified = population_flag.merge(
        customers,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    unified = unified.merge(
        screening,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # Left joins can introduce real NaN values for customers with
    # no matching row (e.g. no screening_results entry). Every
    # source CSV is loaded with fillna("") so downstream control
    # checks compare against "", not NaN. Re-apply it here so the
    # merge doesn't silently reintroduce NaN for unmatched rows.
    unified = unified.fillna("")

    return unified