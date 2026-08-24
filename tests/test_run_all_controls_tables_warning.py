"""
Tests for engine.controls.run_all_controls()'s handling of the
optional `tables` parameter.

Background: run_all_controls(unified) without `tables` used to
silently skip RECON_001 with no error and no warning -- confirmed
during review: 207 findings without `tables`, 223 with it (the missing
16 are RECON_001). The real pipeline (engine.audit_pipeline.run_audit)
always passes `tables`, so this was never a live production risk, but
any direct/manual/notebook call that forgot `tables` would silently
get an incomplete audit with no indication anything was skipped.

`tables` stays optional (tests/test_controls.py has ~11 call sites
that intentionally omit it to test one customer-level control in
isolation -- making it required would force those to pass irrelevant
data). Instead, omitting it now raises a RuntimeWarning so the
omission is visible rather than silent.
"""

import warnings

import pandas as pd
import pytest

from engine.controls import run_all_controls
from engine.data_loader import build_unified_customer_record, load_data


@pytest.fixture(scope="module")
def unified_and_tables():
    tables = load_data()
    unified = build_unified_customer_record(tables)
    return unified, tables


def test_omitting_tables_warns(unified_and_tables):
    unified, _ = unified_and_tables

    with pytest.warns(RuntimeWarning, match="RECON_001"):
        run_all_controls(unified)


def test_omitting_tables_still_returns_customer_level_findings(unified_and_tables):
    """The warning doesn't stop execution -- customer-level controls
    still run normally, since deliberately isolating them is valid."""
    unified, _ = unified_and_tables

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        findings = run_all_controls(unified)

    assert findings
    assert "RECON_001" not in {f["control_id"] for f in findings}


def test_omitting_tables_produces_fewer_findings_than_passing_them(unified_and_tables):
    """Regression guard for the exact gap found in review: 207 vs 223."""
    unified, tables = unified_and_tables

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        without_tables = run_all_controls(unified)

    with_tables = run_all_controls(unified, tables=tables)

    assert len(with_tables) > len(without_tables)
    control_ids_with = {f["control_id"] for f in with_tables}
    control_ids_without = {f["control_id"] for f in without_tables}
    assert control_ids_with - control_ids_without == {"RECON_001"}


def test_passing_tables_does_not_warn(unified_and_tables):
    """Sanity check: the warning is specifically about the omission,
    not raised unconditionally on every call."""
    unified, tables = unified_and_tables

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Should NOT raise, since tables was provided.
        run_all_controls(unified, tables=tables)


def test_empty_tables_dict_does_not_warn(unified_and_tables):
    """
    tables={} is not the same as tables=None -- it's an explicit
    (if unusual) value, so the "did you forget tables?" warning
    should not fire. reconciliation_001 is still called and may raise
    or return oddly on genuinely malformed input; that's a separate
    concern from this warning.
    """
    unified, _ = unified_and_tables

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            run_all_controls(unified, tables={})
        except RuntimeWarning:
            pytest.fail("Should not warn about missing tables when tables={} was given.")
        except Exception:
            # reconciliation_001 may itself fail on an empty tables
            # dict -- that's fine, this test only cares about the
            # warning not firing.
            pass
