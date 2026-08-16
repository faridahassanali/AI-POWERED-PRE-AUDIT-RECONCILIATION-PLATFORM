"""
Layer 1 — Data Ingestion & Normalization.

- load_data(): reads all 8 CSV files from data/ into a dict of DataFrames.
- build_unified_customer_record(): joins the source files into ONE row
  per customer, which is what every control function (Layer 2) will run on.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read every source CSV into a DataFrame. Returns a dict keyed by name."""
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

    tables = {}
    missing = []
    for key, filename in files.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        tables[key] = pd.read_csv(path, dtype=str).fillna("")

    if missing:
        raise FileNotFoundError(
            f"These files are missing from the data/ folder: {missing}. "
            f"Make sure you copied every V3 dataset file into data/."
        )

    return tables


def build_unified_customer_record(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join approved_source_population + customers + screening_results
    into one row per customer_id — this is what every control runs against.

    customers.csv already carries risk_level, screening_status,
    account_status, wallet_status, risk_exception_*, dormant_handling_status.
    We only pull in the extra evidence fields that live in screening_results.csv
    (screening_evidence_present, screening_reference) since controls.json
    lists them as required evidence for SCREENING_001.
    """
    population = tables["approved_source_population"]
    customers = tables["customers"]
    screening = tables["screening_results"][
        ["customer_id", "screening_evidence_present", "screening_reference"]
    ]

    # customers.csv already has name_ar/name_en/national_id, so we only take
    # population_status from approved_source_population to avoid duplicate columns
    population_flag = population[["customer_id", "population_status"]]

    unified = population_flag.merge(customers, on="customer_id", how="left")
    unified = unified.merge(screening, on="customer_id", how="left")

    return unified


if __name__ == "__main__":
    # Quick smoke test — run this file directly to confirm everything works
    tables = load_data()
    print("Files loaded:")
    for name, df in tables.items():
        print(f"  {name}: {len(df)} rows")

    unified = build_unified_customer_record(tables)
    print(f"\nUnified Customer Record: {len(unified)} rows, columns:")
    print(list(unified.columns))
    print("\nSample customer:")
    print(unified.iloc[0].to_dict())
