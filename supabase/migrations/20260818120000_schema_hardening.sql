-- AI-Powered Pre-Audit & Reconciliation Platform
-- Migration: 002_schema_hardening
--
-- Scope:
--   Small, additive follow-up to 001_initial_audit_schema.
--   Does NOT touch any table already in use by the pipeline
--   or the persistence skeleton (audit_runs, findings,
--   finding_reviews). Safe to run on top of 001 with data
--   already in it.
--
-- Contains:
--   1. Drop the duplicate foreign key on findings.audit_run_id
--      (it was declared twice in 001 — once inline on the
--      column, once as a named constraint. Same column, same
--      target, same ON UPDATE/DELETE rules. Harmless but
--      redundant; keeping the named one for a clearer name).
--   2. Add audit_evaluations — stores the ground-truth
--      evaluation metrics (TP/FP/FN, precision/recall/F1) that
--      engine/evaluation_report.py already computes for every
--      audit run, so they can be shown on a dashboard later
--      instead of being recomputed or lost.
--
-- Explicitly NOT included here (deferred, not forgotten):
--   - RLS policies: service_role bypasses RLS regardless of
--     policies, so none are needed until a non-service-role
--     caller (e.g. a frontend using the anon key) needs direct
--     Supabase access. Decide the access model then.
--   - finding_explanations composite FK tightening: out of
--     scope for the current persistence work (audit_runs /
--     findings / finding_reviews only).
-
-- 1. DROP THE DUPLICATE FOREIGN KEY ON findings.audit_run_id


alter table public.findings
    drop constraint if exists findings_audit_run_fk;


-- 2. AUDIT EVALUATIONS


create table if not exists public.audit_evaluations (
    audit_run_id text primary key
        references public.audit_runs(audit_run_id)
        on update cascade
        on delete cascade,

    true_positives integer not null
        check (true_positives >= 0),

    false_positives integer not null
        check (false_positives >= 0),

    false_negatives integer not null
        check (false_negatives >= 0),

    precision numeric(5, 4) not null
        check (precision >= 0 and precision <= 1),

    recall numeric(5, 4) not null
        check (recall >= 0 and recall <= 1),

    f1_score numeric(5, 4) not null
        check (f1_score >= 0 and f1_score <= 1),

    report text,

    created_at timestamptz not null default now()
);

alter table public.audit_evaluations enable row level security;

