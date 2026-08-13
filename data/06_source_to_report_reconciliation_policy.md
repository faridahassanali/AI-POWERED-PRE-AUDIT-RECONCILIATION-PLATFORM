# Source-to-Report Reconciliation Standard
Policy ID: RECON-POLICY-001
Version: 1.0 (Synthetic)

## Requirements
1. Every customer in the approved initialization population should be traceable to the final report.
2. Every report customer should be traceable to the approved source population.
3. Key control fields must match between source/processing data and the final report.
4. Multiple mismatches for the same customer are grouped into one reconciliation finding with all mismatched fields recorded as evidence.
5. Duplicate raw partner requests are deduplicated by customer ID before reconciliation.

## Key Fields
Customer ID, Arabic name, risk level, screening status, account status, wallet status, risk exception approval/reference, and dormant handling status.

## Audit Evidence
The system should retain the source value, report value, mismatch field(s), timestamp, and batch/run identifier.
