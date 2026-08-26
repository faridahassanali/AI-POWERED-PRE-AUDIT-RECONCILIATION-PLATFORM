"""
Database Preflight Verification.

Standalone check run BEFORE deploying/starting the backend, to catch
a misconfigured or half-migrated Supabase database loudly instead of
failing mysteriously later (the PGRST125 class of bug -- a malformed
SUPABASE_URL or an unapplied migration for `policies` /
`policy_versions` is exactly what this is meant to catch early).

Checks, in order:
    1. Connection       -- can we reach Postgres at all?
    2. Tables            -- does every expected table exist?
    3. Columns           -- does every expected table have every
                             expected column?
    4. Constraints        -- primary keys and foreign keys present?
    5. Indexes             -- do the indexes from the migrations exist?
    6. Row Level Security  -- is RLS actually enabled on every table
                             that's supposed to have it?
    7. Grants               -- does service_role have the privileges
                             it needs on every table?

Usage
-----
    export DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    python scripts/verify_database.py

    # or point at a remote/staging project:
    export DATABASE_URL="postgresql://postgres.<ref>:<password>@<pooler-host>:5432/postgres"
    python scripts/verify_database.py

DATABASE_URL must be a direct Postgres connection string (NOT the
Supabase REST/PostgREST URL) -- this script needs pg_catalog /
information_schema access, which the REST API doesn't expose.

For local Supabase dev, the default direct connection is:
    postgresql://postgres:postgres@127.0.0.1:54322/postgres
(port 54322, per supabase/config.toml's [db] section).

Exit code is 0 only if every check passes. Non-zero otherwise, so
this is safe to wire into CI / a pre-deploy step.

Dependency: psycopg2-binary (not in requirements.txt yet -- this
script's own concern, since it's a dev/ops tool, not part of the
runtime app).
    pip install psycopg2-binary --break-system-packages
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print(
        "ERROR: psycopg2 is not installed.\n"
        "Run: pip install psycopg2-binary --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(2)


# =====================================================================
# EXPECTED SCHEMA
#
# Kept in sync with:
#   supabase/migrations/20260817101525_new-migration.sql
#   supabase/migrations/20260818120000_schema_hardening.sql
#   supabase/migrations/20260824110000_service_role_grants.sql
#
# If a migration changes a table's columns/indexes/grants, update
# this dict in the SAME PR -- this script is only useful if it
# actually reflects the migrations.
# =====================================================================

EXPECTED_TABLES: dict[str, list[str]] = {
    "audit_runs": [
        "audit_run_id",
        "started_at",
        "completed_at",
        "controls_executed",
        "total_records_evaluated",
        "total_findings_generated",
        "created_at",
    ],
    "policies": [
        "policy_id",
        "policy_name",
        "description",
        "created_at",
    ],
    "policy_versions": [
        "policy_version_id",
        "policy_id",
        "version",
        "section",
        "policy_text",
        "created_at",
    ],
    "findings": [
        "finding_id",
        "audit_run_id",
        "control_id",
        "customer_id",
        "severity",
        "assessment_status",
        "finding_status",
        "expected",
        "actual",
        "evidence",
        "policy_references",
        "reviewed_by",
        "review_timestamp",
        "reviewer_notes",
        "ai_explanation",
        "ai_recommendation",
        "created_at",
        "updated_at",
    ],
    "finding_explanations": [
        "explanation_id",
        "finding_id",
        "audit_run_id",
        "control_id",
        "customer_id",
        "severity",
        "assessment_status",
        "finding_status",
        "summary",
        "expected_condition",
        "observed_condition",
        "evidence",
        "policy_references",
        "review_action",
        "created_at",
    ],
    "finding_reviews": [
        "review_id",
        "finding_id",
        "audit_run_id",
        "previous_status",
        "new_status",
        "reviewed_by",
        "reviewer_notes",
        "reviewed_at",
    ],
    "ai_outputs": [
        "ai_output_id",
        "finding_id",
        "audit_run_id",
        "ai_explanation",
        "ai_recommendation",
        "model_name",
        "prompt_version",
        "retrieved_policy_context",
        "created_at",
    ],
    "audit_evaluations": [
        "audit_run_id",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
        "report",
        "created_at",
    ],
}

# (table, column) primary keys we expect. Composite/identity PKs are
# checked by presence of a PRIMARY KEY constraint on the table, not
# by exact column match, since identity columns are enough here.
EXPECTED_PRIMARY_KEY_TABLES = set(EXPECTED_TABLES.keys())

# (table, column, references_table) -- foreign keys we expect to exist.
EXPECTED_FOREIGN_KEYS: list[tuple[str, str, str]] = [
    ("policy_versions", "policy_id", "policies"),
    ("findings", "audit_run_id", "audit_runs"),
    ("finding_explanations", "finding_id", "findings"),
    ("finding_explanations", "audit_run_id", "audit_runs"),
    ("finding_reviews", "finding_id", "findings"),
    ("finding_reviews", "audit_run_id", "audit_runs"),
    ("ai_outputs", "finding_id", "findings"),
    ("ai_outputs", "audit_run_id", "audit_runs"),
    ("audit_evaluations", "audit_run_id", "audit_runs"),
]

EXPECTED_INDEXES: list[str] = [
    "idx_findings_audit_run",
    "idx_findings_control",
    "idx_findings_customer",
    "idx_findings_status",
    "idx_findings_severity",
    "idx_finding_reviews_finding",
    "idx_finding_explanations_finding",
    "idx_ai_outputs_finding",
    "idx_policy_versions_policy",
]

EXPECTED_RLS_TABLES = set(EXPECTED_TABLES.keys())

# Privileges service_role must hold on every table, per
# 20260824110000_service_role_grants.sql.
EXPECTED_SERVICE_ROLE_PRIVILEGES = {"SELECT", "INSERT", "UPDATE"}
EXPECTED_SERVICE_ROLE_GRANT_TABLES = set(EXPECTED_TABLES.keys())


# =====================================================================
# RESULT TRACKING
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
        lines.append("DATABASE PREFLIGHT VERIFICATION")
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

def check_connection(conn) -> CheckResult:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return CheckResult("Connection", True, ["Connected successfully."])
    except Exception as exc:
        return CheckResult("Connection", False, [f"Could not connect: {exc}"])


def _fetch_existing_tables(cur) -> set[str]:
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """
    )
    return {row[0] for row in cur.fetchall()}


def check_tables(cur) -> CheckResult:
    existing = _fetch_existing_tables(cur)
    expected = set(EXPECTED_TABLES.keys())
    missing = sorted(expected - existing)

    if missing:
        return CheckResult(
            "Tables",
            False,
            [f"Missing table: {name}" for name in missing],
        )

    return CheckResult(
        "Tables",
        True,
        [f"All {len(expected)} expected tables exist."],
    )


def check_columns(cur) -> CheckResult:
    details: list[str] = []
    all_ok = True

    for table, expected_columns in EXPECTED_TABLES.items():
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s;
            """,
            (table,),
        )
        existing_columns = {row[0] for row in cur.fetchall()}

        if not existing_columns:
            # Table itself missing -- already reported by check_tables.
            continue

        missing = sorted(set(expected_columns) - existing_columns)

        if missing:
            all_ok = False
            details.append(f"{table}: missing column(s) {missing}")

    if all_ok:
        details.append("All expected columns present on all expected tables.")

    return CheckResult("Columns", all_ok, details)


def check_primary_keys(cur) -> CheckResult:
    details: list[str] = []
    all_ok = True

    for table in sorted(EXPECTED_PRIMARY_KEY_TABLES):
        cur.execute(
            """
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_schema = 'public'
              AND table_name = %s
              AND constraint_type = 'PRIMARY KEY';
            """,
            (table,),
        )
        rows = cur.fetchall()

        if not rows:
            all_ok = False
            details.append(f"{table}: no PRIMARY KEY constraint found")

    if all_ok:
        details.append(
            f"All {len(EXPECTED_PRIMARY_KEY_TABLES)} tables have a primary key."
        )

    return CheckResult("Primary Keys", all_ok, details)


def check_foreign_keys(cur) -> CheckResult:
    details: list[str] = []
    all_ok = True

    cur.execute(
        """
        SELECT
            tc.table_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
         AND tc.table_schema = ccu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public';
        """
    )
    existing_fks = {(row[0], row[1], row[2]) for row in cur.fetchall()}

    for table, column, references_table in EXPECTED_FOREIGN_KEYS:
        if (table, column, references_table) not in existing_fks:
            all_ok = False
            details.append(
                f"{table}.{column} -> {references_table}: foreign key not found"
            )

    if all_ok:
        details.append(
            f"All {len(EXPECTED_FOREIGN_KEYS)} expected foreign keys present."
        )

    return CheckResult("Foreign Keys", all_ok, details)


def check_indexes(cur) -> CheckResult:
    cur.execute(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public';
        """
    )
    existing = {row[0] for row in cur.fetchall()}

    missing = sorted(set(EXPECTED_INDEXES) - existing)

    if missing:
        return CheckResult(
            "Indexes",
            False,
            [f"Missing index: {name}" for name in missing],
        )

    return CheckResult(
        "Indexes",
        True,
        [f"All {len(EXPECTED_INDEXES)} expected indexes exist."],
    )


def check_rls_enabled(cur) -> CheckResult:
    cur.execute(
        """
        SELECT c.relname, c.relrowsecurity
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r';
        """
    )
    rows = {row[0]: row[1] for row in cur.fetchall()}

    details: list[str] = []
    all_ok = True

    for table in sorted(EXPECTED_RLS_TABLES):
        rls_enabled = rows.get(table)

        if rls_enabled is None:
            all_ok = False
            details.append(f"{table}: table not found while checking RLS")
        elif not rls_enabled:
            all_ok = False
            details.append(f"{table}: RLS is NOT enabled")

    if all_ok:
        details.append(
            f"RLS is enabled on all {len(EXPECTED_RLS_TABLES)} expected tables."
        )

    return CheckResult("Row Level Security", all_ok, details)


def check_service_role_grants(cur) -> CheckResult:
    cur.execute(
        """
        SELECT table_name, privilege_type
        FROM information_schema.role_table_grants
        WHERE grantee = 'service_role' AND table_schema = 'public';
        """
    )

    grants_by_table: dict[str, set[str]] = {}
    for table_name, privilege_type in cur.fetchall():
        grants_by_table.setdefault(table_name, set()).add(privilege_type)

    details: list[str] = []
    all_ok = True

    for table in sorted(EXPECTED_SERVICE_ROLE_GRANT_TABLES):
        held = grants_by_table.get(table, set())
        missing = EXPECTED_SERVICE_ROLE_PRIVILEGES - held

        if missing:
            all_ok = False
            details.append(
                f"{table}: service_role missing privilege(s) {sorted(missing)}"
            )

    if all_ok:
        details.append(
            "service_role holds SELECT/INSERT/UPDATE on all expected tables."
        )

    return CheckResult("Grants (service_role)", all_ok, details)


# =====================================================================
# MAIN
# =====================================================================

def run_all_checks() -> Report:
    database_url = os.environ.get("DATABASE_URL")

    report = Report()

    if not database_url:
        report.add(
            "Connection",
            False,
            [
                "DATABASE_URL is not set. Example for local Supabase:",
                "  postgresql://postgres:postgres@127.0.0.1:54322/postgres",
            ],
        )
        return report

    try:
        conn = psycopg2.connect(database_url, connect_timeout=10)
    except Exception as exc:
        report.add("Connection", False, [f"Could not connect: {exc}"])
        return report

    try:
        conn_result = check_connection(conn)
        report.results.append(conn_result)

        if not conn_result.passed:
            return report

        with conn.cursor() as cur:
            report.results.append(check_tables(cur))
            report.results.append(check_columns(cur))
            report.results.append(check_primary_keys(cur))
            report.results.append(check_foreign_keys(cur))
            report.results.append(check_indexes(cur))
            report.results.append(check_rls_enabled(cur))
            report.results.append(check_service_role_grants(cur))
    finally:
        conn.close()

    return report


def main() -> int:
    report = run_all_checks()
    print(report.render())
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
