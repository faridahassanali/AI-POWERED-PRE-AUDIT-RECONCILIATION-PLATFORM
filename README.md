# AI-Powered Pre-Audit & Reconciliation Platform

## Automated Self-Audit for Wallet Product Initialization

An AI-assisted pre-audit platform designed to identify potential audit findings **before the formal audit takes place**.

The platform validates wallet initialization data using deterministic audit controls, preserves evidence for every potential finding, reconciles source data with the final report, routes findings through human review, retrieves the applicable policy through RAG, and uses an LLM to generate evidence-backed explanations and recommendations.

> **Core idea:** Audit the process before the auditor audits it.

---

## Table of Contents

- [1. Business Scenario](#1-business-scenario)
- [2. Problem Statement](#2-problem-statement)
- [3. Project Objective](#3-project-objective)
- [4. What the System Does](#4-what-the-system-does)
- [5. MVP Controls](#5-mvp-controls)
- [6. Important Status Model](#6-important-status-model)
- [7. Control Details](#7-control-details)
- [8. Synthetic Dataset](#8-synthetic-dataset)
- [9. Ground Truth](#9-ground-truth)
- [10. Canonical Source Population](#10-canonical-source-population)
- [11. Project Architecture](#11-project-architecture)
- [12. Data Loading and Normalization](#12-data-loading-and-normalization)
- [13. Finding Contract](#13-finding-contract)
- [14. Human Review](#14-human-review)
- [15. Policy Knowledge Base and RAG](#15-policy-knowledge-base-and-rag)
- [16. AI / LLM Role](#16-ai--llm-role)
- [17. End-to-End Example](#17-end-to-end-example)
- [18. Evaluation Strategy](#18-evaluation-strategy)
- [19. Project Structure](#19-project-structure)
- [20. Setup](#20-setup)
- [21. Implementation Roadmap](#21-implementation-roadmap)
- [22. MVP Definition of Done](#22-mvp-definition-of-done)
- [23. Out of Scope](#23-out-of-scope)
- [24. Synthetic Data Integrity Checklist](#24-synthetic-data-integrity-checklist)

---

# 1. Business Scenario

When a bank launches a financial product such as a wallet, a partner company may submit a population of customers for wallet initialization.

The bank is responsible for processing those customers before activation. The process may include:

1. Receiving the partner customer population.
2. Validating customer data.
3. Performing customer screening.
4. Assessing customer risk.
5. Checking the required Arabic customer name.
6. Identifying dormant accounts and performing the required handling.
7. Initializing the wallet.
8. Producing a final report describing the processed population.

The formal audit may later discover that something went wrong.

For example:

- A high-risk customer was opened.
- A customer with an unsuccessful or incomplete screening result was treated as clear.
- A customer does not have an Arabic-script name.
- A dormant account was activated without the required handling.
- The final report does not accurately represent the source or processing data.

The purpose of this project is to identify such problems **before the formal audit**.

---

# 2. Problem Statement

The problem is not only whether a process was performed.

The bank also needs to verify that:

- the required controls were actually satisfied;
- the evidence exists;
- the source and processing data are consistent;
- the final report accurately represents the underlying data;
- the finding can be traced back to the relevant policy;
- a reviewer can understand and confirm or reject the potential finding.

The platform therefore acts as a controlled **pre-audit layer** between operational processing and the formal audit.

---

# 3. Project Objective

The platform aims to:

- Detect potential audit findings automatically.
- Use deterministic rules for compliance decisions.
- Preserve evidence for every potential finding.
- Reconcile the final report against an approved source population.
- Provide a human review stage.
- Retrieve the relevant policy using RAG.
- Use an LLM only for grounded explanations, recommendations, and summaries.
- Produce a readable pre-audit report.

### Design Principle

The LLM should **not decide whether a control failed**.

The deterministic audit engine establishes the finding.

The human reviewer confirms or rejects it.

The RAG + LLM layer then explains confirmed findings using the applicable policy.

---

# 4. What the System Does

The high-level workflow is:

```text
Partner Data
     +
Customer Data
     +
Screening Data
     +
Wallet Processing Data
     +
Final Report
     ↓
Data Loader
     ↓
Normalization
     ↓
Approved Source Population
     ↓
Deterministic Audit Controls
     ↓
Potential Findings
     ↓
Human Review
     ↓
Confirmed Findings
     ↓
Policy Retrieval / RAG
     ↓
LLM
     ↓
Explanation + Recommendation
     ↓
Pre-Audit Report
```

---

# 5. MVP Controls

The MVP contains five deterministic controls.

| Control ID | Control | Main Question | Severity | Policy |
|---|---|---|---|---|
| `SCREENING_001` | Customer Screening Validation | Was mandatory screening clear and evidenced before wallet activation? | Critical / High | `SCREENING-POLICY-001` |
| `RISK_001` | Customer Risk Validation | Was a High Risk customer opened without an approved exception? | High | `RISK-POLICY-001` |
| `ARABIC_NAME_001` | Arabic Name Presence Validation | Does the customer record contain Arabic-script name data? | Medium | `DATA-POLICY-001` |
| `DORMANT_001` | Dormant Account Handling Validation | Was required dormant handling completed before activation? | High | `DORMANT-POLICY-001` |
| `RECON_001` | Source-to-Report Reconciliation | Does the final report match the approved source and processing data? | High | `RECON-POLICY-001` |

---

# 6. Important Status Model

The project separates the **result of a control assessment** from the **lifecycle of the finding**.

```text
assessment_status = PASS | FAIL | UNKNOWN

finding_status =
    REVIEW
    CONFIRMED
    REJECTED
    RESOLVED
```

Example:

```text
Rule Engine
    ↓
assessment_status = FAIL
finding_status = REVIEW
    ↓
Human Review
    ↓
CONFIRMED
```

---

# 7. Control Details

## 7.1 SCREENING_001 — Customer Screening Validation

### Rule

```text
FAIL when:

wallet_status == OPENED
AND
(
    screening_status != CLEAR
    OR
    screening_evidence_present == FALSE
)
```

The synthetic screening policy states that `HIGH_RISK`, `PENDING`, `NO_MATCH`, and missing mandatory screening evidence must not be treated as clear.

### Evidence

- `customer_id`
- `screening_status`
- `screening_evidence_present`
- `screening_reference`
- `wallet_status`

### Severity

- `CRITICAL` when a `HIGH_RISK` customer is opened.
- `HIGH` for other non-CLEAR or missing-mandatory-evidence cases.

---

## 7.2 RISK_001 — Customer Risk Validation

### Rule

```text
FAIL when:

risk_level == HIGH
AND
wallet_status == OPENED
AND
risk_exception_approved == FALSE
```

An approved exception is valid only when the exception is properly recorded.

Relevant fields:

```text
risk_exception_approved
risk_exception_reference
risk_exception_reviewer
```

Therefore:

```text
HIGH + OPENED + exception=False
→ FINDING

HIGH + OPENED + exception=True
→ PASS
```

---

## 7.3 ARABIC_NAME_001 — Arabic Name Presence Validation

### Rule

```text
FAIL when:

name_ar is empty
OR
name_ar contains no Arabic-script character
```

This is intentionally a narrow MVP control.

It validates the **presence of Arabic-script characters**.

It does not claim to validate legal-name correctness, full-name completeness, transliteration equivalence, or identity matching between Arabic and English names.

---

## 7.4 DORMANT_001 — Dormant Account Handling Validation

### Rule

```text
FAIL when:

account_status == DORMANT
AND
wallet_status == OPENED
AND
dormant_handling_status != COMPLETED
```

Here:

- `account_status` = status of the underlying bank account.
- `wallet_status` = status of the newly initialized wallet.

Therefore:

```text
DORMANT + OPENED + handling incomplete
```

is a meaningful potential finding.

---

## 7.5 RECON_001 — Source-to-Report Reconciliation

This control verifies that the final report accurately represents the approved source population and processing results.

```text
Raw Partner Requests
        ↓
Deduplicate by customer_id
        ↓
Approved Source Population
        ↓
Join with Customer Master
        ↓
Enrich with Screening + Wallet Processing
        ↓
Compare with Final Audit Report
```

Relevant fields include:

```text
customer_id
name_ar
risk_level
screening_status
account_status
wallet_status
```

Multiple mismatched fields for the same customer produce **one `RECON_001` finding**. The finding evidence contains all mismatched fields.

---

# 8. Synthetic Dataset

Real bank customer data is not available for the project.

Therefore, the MVP uses a controlled synthetic dataset containing realistic clean records and deliberate exceptions.

The synthetic dataset is designed for:

- development;
- testing;
- demonstration;
- rule-engine evaluation;
- RAG integration;
- end-to-end testing.

| File | Purpose |
|---|---|
| `customers.csv` | Customer master records |
| `partner_wallet_requests.csv` | Raw partner requests, including deliberate duplicates |
| `approved_source_population.csv` | Canonical source population used by the audit engine |
| `screening_results.csv` | Screening results and evidence |
| `wallet_initialization.csv` | Wallet initialization results |
| `dormant_accounts_report.csv` | Dormant-account records and required handling |
| `final_wallet_audit_report.csv` | Final report to be reconciled |
| `expected_findings.csv` | Ground-truth findings |
| `controls.json` | Machine-readable control contract |
| `finding_schema.json` | Finding and review lifecycle contract |
| `01_customer_screening_policy.md` | Screening policy |
| `02_risk_management_policy.md` | Risk policy |
| `03_arabic_name_data_policy.md` | Arabic-name policy |
| `04_dormant_accounts_policy.md` | Dormant-account policy |
| `05_product_initialization_policy.md` | Product initialization policy |
| `06_source_to_report_reconciliation_policy.md` | Reconciliation policy |

> The exact record counts and ground-truth distribution should be taken from the V3 dataset and treated as the test baseline. Do not manually change them while evaluating the engine.

---

# 9. Ground Truth

`expected_findings.csv` is the reference set used to evaluate the deterministic audit engine.

It represents the findings intentionally injected into the synthetic dataset.

The evaluation process is:

```text
Synthetic V3 Dataset
        ↓
Audit Engine
        ↓
Generated Findings
        ↓
Compare with expected_findings.csv
        ↓
Precision / Recall / F1
```

The ground truth should be treated as an **evaluation oracle**, not as an input to the production rule engine.

---

# 10. Canonical Source Population

The raw partner file may contain duplicate requests.

The approved source population is the normalized population against which downstream processing and final reporting are evaluated.

Conceptually:

```text
Raw Partner Requests
        ↓
Validation
        ↓
Deduplication by customer_id
        ↓
Approved Source Population
```

The canonical record is then enriched with:

```text
Customer Master
+
Screening Result
+
Wallet Initialization
+
Account Status
```

This prevents the reconciliation control from comparing the final report against an ambiguous raw input.

---

# 11. Project Architecture

```text
                         INPUT DATA
                              │
                 ┌────────────┼────────────┐
                 │            │            │
             Partner      Customer      Processing
              Data         Master       Data / Report
                 └────────────┼────────────┘
                              ↓
                    DATA LOADER + NORMALIZER
                              ↓
                    APPROVED SOURCE POPULATION
                              ↓
                       AUDIT CONTROL ENGINE
                              │
          ┌───────────────────┼───────────────────┐
          ↓                   ↓                   ↓
     Screening              Risk              Arabic Name
          ↓                   ↓                   ↓
       Dormant         Reconciliation        Other Controls
          └───────────────────┬───────────────────┘
                              ↓
                         POTENTIAL FINDINGS
                              ↓
                         HUMAN REVIEW
                    ┌─────────┴─────────┐
                    ↓                   ↓
                CONFIRMED            REJECTED
                    │
                    ↓
              POLICY KNOWLEDGE BASE
                    ↓
                   RAG
                    ↓
                   LLM
                    ↓
          Explanation / Recommendation
                    ↓
               PRE-AUDIT REPORT
```

---

# 12. Data Loading and Normalization

The first implementation stage is the data foundation.

Before implementing RAG or the LLM, the project must be able to load and normalize the V3 dataset reliably.

Responsibilities:

1. Load all required CSV files.
2. Validate required columns.
3. Validate basic data types.
4. Normalize status values.
5. Normalize Boolean values.
6. Normalize dates.
7. Normalize missing values.
8. Normalize identifiers without changing their business meaning.
9. Deduplicate partner requests.
10. Generate the approved source population.
11. Produce canonical records for the audit controls.

First implementation:

```text
engine/load_data.py
```

followed by normalization.

---

# 13. Finding Contract

Example:

```json
{
  "finding_id": "F-0001",
  "audit_run_id": "RUN-2026-08-13-001",
  "control_id": "RISK_001",
  "customer_id": "CUST100002",
  "severity": "HIGH",
  "assessment_status": "FAIL",
  "finding_status": "REVIEW",
  "expected": "...",
  "actual": "...",
  "evidence": {},
  "policy_references": [
    {
      "policy_id": "RISK-POLICY-001",
      "version": "1.0",
      "section": "Requirements"
    }
  ],
  "reviewed_by": null,
  "review_timestamp": null,
  "reviewer_notes": null,
  "ai_explanation": null,
  "ai_recommendation": null
}
```

The AI fields are initially empty.

---

# 14. Human Review

The rule engine produces potential findings.

A human reviewer is the control gate before AI explanation.

```text
Rule Engine
    ↓
assessment_status = FAIL
finding_status = REVIEW
    ↓
Reviewer examines evidence
    ├── CONFIRMED
    └── REJECTED
          ↓
     Confirmed findings
          ↓
         RAG
          ↓
         LLM
```

The reviewer should see:

- customer;
- control;
- severity;
- expected value;
- actual value;
- evidence;
- policy reference.

---

# 15. Policy Knowledge Base and RAG

The project contains synthetic policy documents covering the five audit areas and the supporting product-initialization process.

Pipeline:

```text
Policy Documents
        ↓
Text Extraction
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector Store
        ↓
Retriever
        ↓
Relevant Policy / Section
```

The RAG layer retrieves:

```text
Policy ID
Policy Version
Policy Section
Policy Text
```

The policy reference is stored with the finding.

---

# 16. AI / LLM Role

AI is included, but its role is intentionally constrained.

| Task | Responsible Layer |
|---|---|
| Decide whether a control failed | Deterministic Rule Engine |
| Calculate finding severity | Deterministic Rule Engine |
| Confirm / reject a finding | Human Reviewer |
| Retrieve applicable policy | RAG |
| Explain a confirmed finding | LLM |
| Generate recommendation | LLM |
| Calculate statistics | Deterministic code |
| Write executive summary | LLM using calculated statistics |
| Autonomous remediation | Not allowed |

The LLM must not invent customers, evidence, control failures, severity, policy requirements, or statistics.

---

# 17. End-to-End Example

Suppose:

```text
Customer: CUST100002
Risk: HIGH
Wallet: OPENED
Approved Exception: FALSE
```

The rule engine evaluates `RISK_001`:

```text
assessment_status = FAIL
finding_status = REVIEW
severity = HIGH
```

The reviewer examines the evidence.

If confirmed:

```text
finding_status = CONFIRMED
```

Then:

```text
Confirmed Finding
       ↓
RAG
       ↓
RISK-POLICY-001
       ↓
Relevant Policy Section
       ↓
LLM
```

The LLM generates an explanation and recommendation based on the confirmed finding, evidence, and retrieved policy.

---

# 18. Evaluation Strategy

## 18.1 Deterministic Rule Engine

Compare generated findings with:

```text
expected_findings.csv
```

Track:

- Precision
- Recall
- F1
- False Positives
- False Negatives
- Findings per control
- Findings by severity

### Precision

```text
Correct Generated Findings
--------------------------
All Generated Findings
```

### Recall

```text
Correct Generated Findings
--------------------------
All Expected Findings
```

### F1

```text
2 × Precision × Recall
----------------------
Precision + Recall
```

## 18.2 Finding Identity

For evaluation, a finding should be matched using a stable identity such as:

```text
control_id + customer_id
```

For controls where a finding is not customer-specific, use the appropriate control-level key.

Evidence differences should be evaluated separately from finding identity.

## 18.3 RAG Evaluation

Check:

- correct policy;
- correct policy version;
- correct section;
- whether the retrieved context supports the finding.

## 18.4 LLM Evaluation

Check:

- grounding;
- factual consistency;
- relevance;
- completeness;
- recommendation quality;
- absence of unsupported claims.

---

# 19. Project Structure

```text
wallet_audit_project/
│
├── data/
│   ├── customers.csv
│   ├── partner_wallet_requests.csv
│   ├── approved_source_population.csv
│   ├── screening_results.csv
│   ├── wallet_initialization.csv
│   ├── dormant_accounts_report.csv
│   ├── final_wallet_audit_report.csv
│   ├── expected_findings.csv
│   ├── controls.json
│   ├── finding_schema.json
│   └── *.md
│
├── engine/
│   ├── load_data.py
│   ├── normalizer.py
│   ├── audit_engine.py
│   └── controls/
│       ├── screening.py
│       ├── risk.py
│       ├── arabic_name.py
│       ├── dormant.py
│       └── reconciliation.py
│
├── rag/
│   ├── ingestion.py
│   ├── embeddings.py
│   └── retriever.py
│
├── ai/
│   └── report_generator.py
│
├── tests/
│   ├── test_loader.py
│   ├── test_screening.py
│   ├── test_risk.py
│   ├── test_arabic_name.py
│   ├── test_dormant.py
│   └── test_reconciliation.py
│
├── requirements.txt
├── SETUP.md
└── README.md
```

---

# 20. Setup

## Prerequisites

Recommended:

```text
Python 3.10+
```

## Step 1 — Open the Project

Extract the project and open a terminal inside:

```text
wallet_audit_project/
```

You should see:

```text
wallet_audit_project/
├── data/
├── engine/
├── tests/
├── requirements.txt
├── SETUP.md
└── README.md
```

## Step 2 — Create and Activate a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

The initial dependencies include:

```text
pandas
jsonschema
```

Verify:

```bash
python -c "import pandas; import jsonschema; print('ready')"
```

Expected:

```text
ready
```

## Step 4 — Add the V3 Synthetic Dataset

Copy **all V3 files** into:

```text
data/
```

Expected structure:

```text
data/
├── customers.csv
├── partner_wallet_requests.csv
├── approved_source_population.csv
├── screening_results.csv
├── wallet_initialization.csv
├── dormant_accounts_report.csv
├── final_wallet_audit_report.csv
├── expected_findings.csv
├── controls.json
├── finding_schema.json
├── 01_customer_screening_policy.md
├── 02_risk_management_policy.md
├── 03_arabic_name_data_policy.md
├── 04_dormant_accounts_policy.md
├── 05_product_initialization_policy.md
└── 06_source_to_report_reconciliation_policy.md
```

> Do not manually modify the V3 dataset while evaluating the engine. It is also the ground truth for testing.

---

# 21. Implementation Roadmap

## Phase 1 — Data Foundation

```text
Data Loader
    ↓
Schema Validation
    ↓
Normalization
    ↓
Approved Source Population
```

First implementation:

```text
engine/load_data.py
```

## Phase 2 — Deterministic Controls

Implement:

```text
1. SCREENING_001
2. RISK_001
3. ARABIC_NAME_001
4. DORMANT_001
5. RECON_001
```

Start with:

```text
SCREENING_001
```

## Phase 3 — Ground Truth Evaluation

Compare:

```text
Generated Findings
        vs
expected_findings.csv
```

## Phase 4 — Human Review

Implement:

```text
REVIEW
CONFIRMED
REJECTED
RESOLVED
```

with reviewer metadata.

## Phase 5 — RAG

```text
Policy ingestion
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retrieval
```

## Phase 6 — LLM

Use the LLM for:

- confirmed-finding explanations;
- policy-grounded recommendations;
- executive summaries;
- report wording.

## Phase 7 — Reporting

Produce a pre-audit report containing:

- Audit summary
- Overall risk
- Findings
- Severity
- Evidence
- Policy references
- Recommendations
- Review status
- Control statistics

---

# 22. MVP Definition of Done

- [ ] All V3 input files can be loaded.
- [ ] Required schemas are validated.
- [ ] Data normalization works.
- [ ] The approved source population is generated deterministically.
- [ ] All five controls run without an LLM.
- [ ] Findings contain evidence.
- [ ] Findings contain policy references.
- [ ] The engine can be evaluated against the V3 ground-truth findings.
- [ ] Human reviewers can confirm or reject findings.
- [ ] RAG retrieves the relevant policy/version.
- [ ] The LLM generates a grounded explanation.
- [ ] The LLM generates an evidence-backed recommendation.
- [ ] A final pre-audit report can be generated.

---

# 23. Out of Scope

The MVP does not aim to:

- Replace the formal audit function.
- Make autonomous compliance decisions through an LLM.
- Make AI-based risk decisions.
- Perform AI root-cause analysis.
- Use an LLM to decide finding classification when the failed control already identifies the category.
- Perform autonomous remediation.
- Require continuous monitoring.
- Use real customer PII.

---

# 24. Synthetic Data Integrity Checklist

The V3 baseline should be kept internally consistent and version-controlled.

Before starting implementation, verify:

- [ ] Customer IDs are consistent across source files.
- [ ] Partner duplicates are intentional and documented.
- [ ] The approved source population is reproducible.
- [ ] Screening records map to the expected customers.
- [ ] Wallet records map to the expected customers.
- [ ] Dormant records use the agreed account-status values.
- [ ] Risk exceptions are explicitly represented where applicable.
- [ ] Dormant handling status is explicitly represented where applicable.
- [ ] Final-report records use the same canonical identifiers.
- [ ] Reconciliation differences are intentional.
- [ ] `expected_findings.csv` matches the intended injected exceptions.
- [ ] Policy IDs and versions match the control contract.
- [ ] No real customer PII is present.

---

# Final Project Statement

> **An AI-assisted pre-audit platform that automatically checks wallet initialization data against deterministic compliance controls, reconciles the approved source population with the final report, routes potential findings through human review, retrieves the applicable policy through RAG, and uses an LLM to turn confirmed findings into evidence-backed explanations and recommendations.**

The implementation starts with:

```text
V3 Synthetic Data
       ↓
load_data()
       ↓
Normalization
       ↓
SCREENING_001
       ↓
Generated Findings
       ↓
expected_findings.csv
```

Only after the deterministic foundation is working reliably do we add the RAG and LLM layers.
