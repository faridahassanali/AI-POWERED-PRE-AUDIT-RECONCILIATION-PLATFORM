# Synthetic Wallet Pre-Audit Dataset v3

Synthetic development/test data only. No real bank or customer data.

## Purpose
This dataset simulates a wallet product initialization process and is designed to test a deterministic pre-audit engine before adding RAG and LLM capabilities.

## Five deterministic controls
1. SCREENING_001 — customer screening and mandatory evidence
2. RISK_001 — customer risk and approved exceptions
3. ARABIC_NAME_001 — Arabic-script name presence
4. DORMANT_001 — dormant handling before wallet activation
5. RECON_001 — source-to-final-report reconciliation

## Assessment vs. finding lifecycle
- `assessment_status`: PASS / FAIL / UNKNOWN — result of the control assessment.
- `finding_status`: REVIEW / CONFIRMED / REJECTED / RESOLVED — lifecycle after a potential finding is generated.
- A rule engine can generate `assessment_status=FAIL` with `finding_status=REVIEW`; a human reviewer then confirms or rejects it.

## Canonical source population
`approved_source_population.csv` contains one row per distinct partner customer requested for the wallet and present in the bank customer master. Duplicate partner requests are retained in the raw partner file for data-quality testing but are deduplicated by `customer_id` for reconciliation. Partner-only IDs are excluded from the approved population because they have not passed the mandatory customer-master validation stage.

## Main files
- customers.csv
- partner_wallet_requests.csv
- approved_source_population.csv
- screening_results.csv
- wallet_initialization.csv
- dormant_accounts_report.csv
- final_wallet_audit_report.csv
- expected_findings.csv
- controls.json
- finding_schema.json
- six synthetic policy documents

## AI boundary
Deterministic controls decide whether a finding exists. Human review confirms or rejects findings. RAG retrieves the relevant policy version/section. The LLM explains confirmed findings and creates grounded summaries/recommendations. The LLM does not decide whether a customer is compliant.

## Ground truth
`expected_findings.csv` contains 223 expected control findings. It is the evaluation oracle for the deterministic audit engine. The definition is one finding per control violation per customer/record; multiple field mismatches for one customer under RECON_001 are grouped into one finding with all mismatched fields in evidence.
