AI-POWERED PRE-AUDIT & RECONCILIATION PLATFORM
Automated Self-Audit for Wallet Product Initialization
Project Proposal — Final MVP Specification
Business Case • Data Contract • Controls • Human Review • RAG • LLM • Evaluation


## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Business Scenario](#2-business-scenario)
- [3. Objective](#3-objective)
- [4. MVP Controls](#4-mvp-controls)
- [5. Important Status Model](#5-important-status-model)
- [6. SCREENING_001 — Customer Screening Validation](#6-screening001--customer-screening-validation)
- [7. RISK_001 — Customer Risk Validation](#7-risk001--customer-risk-validation)
- [8. ARABIC_NAME_001 — Arabic Name Presence Validation](#8-arabicname001--arabic-name-presence-validation)
- [9. DORMANT_001 — Dormant Account Handling Validation](#9-dormant001--dormant-account-handling-validation)
- [10. RECON_001 — Source-to-Report Reconciliation](#10-recon001--source-to-report-reconciliation)
- [11. Synthetic Dataset v3](#11-synthetic-dataset-v3)
- [12. Ground Truth Distribution](#12-ground-truth-distribution)
- [13. Canonical Source Population](#13-canonical-source-population)
- [14. Overall Architecture](#14-overall-architecture)
- [15. Data Loader & Normalization](#15-data-loader--normalization)
- [16. Canonical Customer Record](#16-canonical-customer-record)
- [17. Finding Contract](#17-finding-contract)
- [18. Human Review](#18-human-review)
- [19. Policy Knowledge Base & RAG](#19-policy-knowledge-base--rag)
- [20. AI Role](#20-ai-role)
- [21. End-to-End Example](#21-end-to-end-example)
- [22. Evaluation Strategy](#22-evaluation-strategy)
- [23. Implementation Roadmap](#23-implementation-roadmap)
- [24. Proposed Backend Structure](#24-proposed-backend-structure)
- [25. MVP Definition of Done](#25-mvp-definition-of-done)
- [26. Out of Scope](#26-out-of-scope)
- [27. Final Project Statement](#27-final-project-statement)
- [28. Synthetic Data Integrity Checklist](#28-synthetic-data-integrity-checklist)
- [Tables and Structured Specifications](#tables-and-structured-specifications)
## 1. Executive Summary
When a bank launches a financial product such as a wallet, a partner company may submit a customer population for wallet initialization. Before activation, the bank performs screening, risk checks, customer-data validation, dormant-account handling, wallet initialization, and final reporting.
The operational problem is not only whether each activity was performed. The bank also needs to know whether the resulting data is complete, consistent, supported by evidence, and accurately represented in the final report. A formal audit may discover an issue only after the wallet population has already been processed.
The proposed platform acts as a controlled pre-audit layer. It runs deterministic audit controls over the source and processing data, creates traceable findings, sends findings through human review, retrieves the applicable policy through RAG, and uses an LLM to explain confirmed findings and produce evidence-backed recommendations.
Core idea: Audit the process before the auditor audits it.

## 2. Business Scenario
Partner Company -> Customer Population -> Bank Processing -> Screening / Risk / Data / Dormancy -> Wallet Initialization -> Final Report -> Formal Audit
A typical audit issue could be: a customer appears as High Risk but has an opened wallet, a customer has a non-clear screening result but the final report shows CLEAR, a required Arabic name is missing, a dormant account was activated without completed handling, or a customer/report record is missing or inconsistent.

## 3. Objective
The platform is designed to identify potential audit findings before the formal audit, preserve the evidence that caused each finding, connect the finding to the relevant policy, and make confirmed findings understandable through AI.
- Detect control violations deterministically.
- Prevent the LLM from making the compliance decision.
- Give a reviewer a clear evidence package for every potential finding.
- Reconcile the final report against an approved source population.
- Use RAG to ground AI explanations in the applicable policy version.
- Generate a readable pre-audit report from structured results.

## 4. MVP Controls

## 5. Important Status Model
The project deliberately separates the result of a control assessment from the lifecycle of the finding. This prevents terms such as FAILED and CONFIRMED from being treated as the same concept.
assessment_status = PASS | FAIL | UNKNOWN
finding_status    = REVIEW | CONFIRMED | REJECTED | RESOLVED

Example:
Rule Engine -> assessment_status = FAIL
             finding_status    = REVIEW
Human Review -> CONFIRMED or REJECTED
The synthetic policies define mandatory evidence for the current controls. Therefore, missing mandatory screening evidence is a FAIL under SCREENING_001. UNKNOWN remains available in the contract for future controls where evidence is genuinely unavailable and the policy does not establish a mandatory requirement.

## 6. SCREENING_001 — Customer Screening Validation
FAIL when wallet_status == OPENED AND (screening_status != CLEAR OR screening_evidence_present == FALSE)
The synthetic screening policy explicitly states that HIGH_RISK, PENDING, NO_MATCH, and missing mandatory screening evidence must not be treated as clear.
Evidence:
- customer_id
- screening_status
- screening_evidence_present
- screening_reference
- wallet_status
Severity:
- CRITICAL when a HIGH_RISK customer is opened.
- HIGH for other non-CLEAR or missing-mandatory-evidence cases.

## 7. RISK_001 — Customer Risk Validation
FAIL when risk_level == HIGH AND wallet_status == OPENED AND risk_exception_approved == FALSE
An approved exception is a legitimate pass for this control only when the exception is recorded with an exception reference and reviewer. This prevents the rule from treating every High Risk customer as an automatic violation.

## 8. ARABIC_NAME_001 — Arabic Name Presence Validation
FAIL when name_ar is empty OR name_ar contains no Arabic-script character
This is intentionally a narrow MVP control. It validates the presence of Arabic-script characters; it does not claim to validate legal-name correctness, completeness, or transliteration equivalence.

## 9. DORMANT_001 — Dormant Account Handling Validation
FAIL when account_status == DORMANT AND wallet_status == OPENED AND dormant_handling_status != COMPLETED
Here, account_status refers to the underlying bank account, while wallet_status refers to the newly initialized wallet. Therefore, DORMANT + OPENED is a meaningful combination: it means a wallet was activated for an underlying dormant account.

## 10. RECON_001 — Source-to-Report Reconciliation
This control checks whether the final report represents the approved source population and processing results accurately.
Raw Partner Requests
        v
Deduplicate by customer_id
        v
Join with Customer Master
        v
Approved Source Population
        v
Enrich with Screening + Wallet Processing
        v
Compare with Final Audit Report
Multiple field mismatches for one customer produce one RECON_001 finding. The finding evidence contains all mismatched fields.

## 11. Synthetic Dataset v3
Real bank customer data is not available for the project, so the MVP uses synthetic development/test data. The data intentionally contains realistic clean records plus controlled exceptions that serve as the evaluation oracle.

## 12. Ground Truth Distribution
Total expected findings: 223. Severity distribution: 7 Critical, 205 High, 11 Medium.

## 13. Canonical Source Population
The raw partner file contains three duplicate customer requests to test deduplication. It no longer contains partner-only IDs that would bypass the defined customer-master validation stage.
approved_source_population =
    distinct partner customer_id
    WHERE requested_wallet = YES
    AND customer_id exists in customers.csv
This produces 1,000 canonical source customers. Duplicate raw partner rows remain visible in the raw input for data-quality testing but do not create duplicate reconciliation findings.

## 14. Overall Architecture
INPUT DATA
                              │
                 ┌────────────┼────────────┐
                 │            │            │
             Partner      Customer      Processing
              Data         Master       Data / Report
                 `--──────────┼────────────┘
                              v
                    DATA LOADER + NORMALIZER
                              v
                    APPROVED SOURCE POPULATION
                              v
                       AUDIT CONTROL ENGINE
                              │
          ┌───────────────────┼───────────────────┐
          v                   v                   v
     Screening              Risk              Arabic Name
          v                   v                   v
       Dormant         Reconciliation      [other controls]
          `--─────────────────┬───────────────────┘
                              v
                           FINDINGS
                              v
                        HUMAN REVIEW
                    ┌─────────┴─────────┐
                    v                   v
                CONFIRMED            REJECTED
                    │
                    v
              POLICY KNOWLEDGE BASE
                    v
                   RAG
                    v
                   LLM
                    v
          Explanation / Recommendation
                    v
               AUDIT REPORT

## 15. Data Loader & Normalization
The first implementation step is not RAG. It is a stable data contract.
- Load all CSV files with explicit schemas.
- Validate required columns and basic types.
- Normalize status strings, Boolean values, dates and missing values.
- Normalize identifiers without changing their business meaning.
- Deduplicate partner requests by customer_id for the approved population.
- Join source and processing data into a canonical record for controls.

## 16. Canonical Customer Record
{
  "customer_id": "CUST100001",
  "name_ar": "ياسمين عمر",
  "name_en": "Yasmine Omar",
  "risk_level": "LOW",
  "screening_status": "CLEAR",
  "screening_evidence_present": true,
  "screening_reference": "SCR-CUST100001",
  "account_status": "ACTIVE",
  "wallet_status": "OPENED",
  "risk_exception_approved": false,
  "risk_exception_reference": null,
  "risk_exception_reviewer": null,
  "dormant_handling_status": "NOT_REQUIRED"
}

## 17. Finding Contract
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
  "evidence": {...},
  "policy_references": [
    {"policy_id": "RISK-POLICY-001", "version": "1.0", "section": "Requirements"}
  ],
  "reviewed_by": null,
  "review_timestamp": null,
  "reviewer_notes": null,
  "ai_explanation": null,
  "ai_recommendation": null
}

## 18. Human Review
The rule engine produces potential findings. A human reviewer is the control gate before AI narrative generation.
Rule Engine
   v
assessment_status = FAIL
finding_status = REVIEW
   v
Reviewer examines evidence
   |-- CONFIRMED
   `-- REJECTED
          v
     confirmed findings
          v
         RAG
          v
         LLM
This design prevents the LLM from deciding whether a customer is compliant and creates a clear audit trail for who reviewed the finding and when.

## 19. Policy Knowledge Base & RAG
Six synthetic policies -> text extraction -> chunking -> embeddings -> vector store -> retrieval
The RAG layer is policy-aware. A confirmed finding retrieves the relevant policy ID, version and section. The retrieved text becomes context for the LLM.
Policy references are stored with version information so that an audit finding remains traceable to the policy version used at the time.

## 20. AI Role
The AI layer is intentionally constrained: it explains and summarizes facts that have already been established by deterministic controls and human review.

## 21. End-to-End Example
Customer: CUST100002
Risk: HIGH
Wallet: OPENED
Approved Exception: FALSE

        v

RISK_001
Assessment: FAIL
Finding: REVIEW

        v

Human Review
        v
CONFIRMED

        v

RAG retrieves RISK-POLICY-001 v1.0

        v

LLM:
Explains the confirmed violation using the evidence
and policy, then recommends the appropriate review/
remediation action.
The LLM does not create RISK_001 and does not change its severity. It receives the structured finding and policy context.

## 22. Evaluation Strategy
The deterministic audit engine is evaluated against expected_findings.csv. The dataset is intentionally constructed so that each control has known exceptions.
- Precision = correct generated findings / all generated findings.
- Recall = correct generated findings / all expected findings.
- F1 = harmonic mean of precision and recall.
- False Positives = generated findings not present in ground truth.
- False Negatives = expected findings not generated by the engine.
For RAG, evaluation should check whether the retrieved policy is the correct policy/version and whether the relevant section is present. For the LLM, evaluation should focus on grounding, factual consistency, relevance, and recommendation quality.

## 23. Implementation Roadmap
Phase 1 — Data foundation: Data Loader -> Schema validation -> Normalization -> Approved Source Population
Phase 2 — Deterministic controls: SCREENING_001 -> RISK_001 -> ARABIC_NAME_001 -> DORMANT_001 -> RECON_001
Phase 3 — Ground-truth evaluation: Compare generated findings with 223 expected findings; calculate precision/recall/F1
Phase 4 — Human Review: Review queue -> Confirm / Reject -> reviewer metadata and audit trail
Phase 5 — RAG: Ingest six policies -> chunk -> embed -> retrieve by control/policy
Phase 6 — LLM: Generate grounded explanations, recommendations and summaries for confirmed findings
Phase 7 — Reporting: Dashboard/API -> findings -> evidence -> policy references -> pre-audit report

## 24. Proposed Backend Structure
backend/
|-- app/
│   |-- main.py
│   |-- data/
│   │   |-- loader.py
│   │   |-- schemas.py
│   │   `-- normalizer.py
│   |-- audit/
│   │   |-- engine.py
│   │   `-- controls/
│   │       |-- screening.py
│   │       |-- risk.py
│   │       |-- arabic_name.py
│   │       |-- dormant.py
│   │       `-- reconciliation.py
│   |-- review/
│   │   `-- findings.py
│   |-- rag/
│   │   |-- ingestion.py
│   │   |-- retriever.py
│   │   `-- embeddings.py
│   |-- ai/
│   │   `-- report_generator.py
│   `-- models/
│       `-- finding.py
|-- data/
|-- policies/
`-- tests/

## 25. MVP Definition of Done
- All seven input/data artifacts can be loaded and normalized.
- The approved source population is generated deterministically.
- All five controls run without relying on an LLM.
- The engine can reproduce the 223 ground-truth findings.
- Each finding contains evidence and policy references.
- A human can confirm or reject a finding.
- RAG retrieves the correct policy/version for a confirmed finding.
- The LLM produces a grounded explanation and recommendation.
- A final pre-audit report can summarize findings, severity, evidence and policy references.

## 26. Out of Scope
- Replacing the formal audit function.
- Autonomous compliance decisions by an LLM.
- AI-based risk scoring.
- AI root-cause analysis.
- AI-based finding classification when the failed control already identifies the category.
- Autonomous remediation.
- Continuous monitoring as an MVP requirement.
- Use of real customer PII.

## 27. Final Project Statement
An AI-assisted pre-audit platform that automatically checks wallet initialization data against deterministic compliance controls, reconciles the approved source population with the final report, routes potential findings through human review, retrieves the applicable policy through RAG, and uses an LLM to turn confirmed findings into evidence-backed explanations and recommendations.

## 28. Synthetic Data Integrity Checklist
- 1,000 customer master records.
- 1,003 raw partner requests with exactly 1,000 unique customers and 3 deliberate duplicate requests.
- 1,000 approved source-population records after deduplication and customer-master validation.
- 1,000 screening records.
- 1,000 wallet initialization records.
- 78 dormant accounts.
- 996 final-report records.
- 223 ground-truth findings.
- No partner-only IDs remain in the raw partner input.
- Approved High Risk exceptions are represented with reference/reviewer and are not treated as RISK_001 findings.
- Dormant findings require incomplete handling, while completed dormant handling passes.
- Reconciliation produces exactly 16 customer-level findings under the canonical source definition.

## Tables and Structured Specifications

| Control ID | Control | Main Question | Severity | Policy |
| --- | --- | --- | --- | --- |
| SCREENING_001 | Customer Screening Validation | Was mandatory screening clear and evidenced before wallet activation? | CRITICAL / HIGH | SCREENING-POLICY-001 |
| RISK_001 | Customer Risk Validation | Was a High Risk customer opened without an approved exception? | HIGH | RISK-POLICY-001 |
| ARABIC_NAME_001 | Arabic Name Presence Validation | Does the customer record contain Arabic-script name data? | MEDIUM | DATA-POLICY-001 |
| DORMANT_001 | Dormant Account Handling Validation | Was required dormant handling completed before activation? | HIGH | DORMANT-POLICY-001 |
| RECON_001 | Source-to-Report Reconciliation | Does the final report match the approved source and processing data? | HIGH | RECON-POLICY-001 |

| File | Purpose / Size |
| --- | --- |
| customers.csv | 1,000 customer master records; risk, screening, account, wallet, exception and dormant fields. |
| partner_wallet_requests.csv | 1,003 raw partner rows: 1,000 unique customers + 3 duplicate requests. |
| approved_source_population.csv | 1,000 canonical customers after partner/customer validation and deduplication. |
| screening_results.csv | 1,000 screening records with status, date, reference and evidence flag. |
| wallet_initialization.csv | 1,000 wallet initialization records with status, exceptions and batch metadata. |
| dormant_accounts_report.csv | 78 dormant-account records. |
| final_wallet_audit_report.csv | 996 report records, intentionally containing controlled reconciliation exceptions. |
| expected_findings.csv | 223 ground-truth control findings. |
| controls.json | Machine-readable control contract. |
| finding_schema.json | Machine-readable finding and review lifecycle contract. |
| Six policy .md files | Synthetic policy knowledge base used by RAG. |

| Control | Expected Findings | Primary Severity |
| --- | --- | --- |
| SCREENING_001 | 127 | 7 Critical, 120 High |
| RISK_001 | 8 | High |
| ARABIC_NAME_001 | 11 | Medium |
| DORMANT_001 | 61 | High |
| RECON_001 | 16 | High |

| Task | Deterministic / Human | AI |
| --- | --- | --- |
| Decide whether a control failed | Rule Engine | No |
| Confirm/reject a potential finding | Human Reviewer | No |
| Retrieve relevant policy | — | RAG |
| Explain a confirmed finding | — | LLM |
| Generate evidence-backed recommendation | — | LLM |
| Generate executive summary from calculated statistics | Rule Engine calculates; human can review | LLM writes |
| Autonomously remediate | No | No |
