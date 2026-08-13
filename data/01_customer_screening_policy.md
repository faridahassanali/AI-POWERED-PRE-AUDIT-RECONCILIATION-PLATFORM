# Customer Screening & Wallet Activation Policy
Policy ID: SCREENING-POLICY-001
Version: 1.0 (Synthetic)

## Purpose
Define the minimum screening control before activating a wallet.

## Requirements
1. Every customer must have a screening record before wallet activation.
2. A customer with screening status `CLEAR` may proceed, subject to other controls.
3. `HIGH_RISK`, `PENDING`, `NO_MATCH`, or missing mandatory screening evidence must not be treated as clear.
4. A wallet must not be activated while mandatory screening evidence is unresolved.

## Audit Evidence
The audit trail should retain customer ID, screening date, screening status, screening reference, screening evidence flag, and wallet activation status.
