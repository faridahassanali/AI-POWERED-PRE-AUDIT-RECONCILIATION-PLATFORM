
-- AI-Powered Pre-Audit & Reconciliation Platform
-- Migration: 001_initial_audit_schema
--
-- Scope:
--   Core persistence schema for the Pre-Audit platform.
--
-- This migration does NOT contain:
--   - HerAI tables
--   - frontend-specific tables
--   - RAG vector implementation
--   - LLM implementation
--   - operational source CSV tables
--
-- Design principle:
--   Deterministic audit engine -> findings
--   Human review -> confirmed/rejected
--   RAG + AI -> explanation/recommendation


-- 1. AUDIT RUNS

create table if not exists public.audit_runs (
    audit_run_id text primary key,

    started_at timestamptz not null default now(),

    completed_at timestamptz,

    controls_executed text[] not null default '{}',

    total_records_evaluated integer not null default 0
        check (total_records_evaluated >= 0),

    total_findings_generated integer not null default 0
        check (total_findings_generated >= 0),

    created_at timestamptz not null default now()
);


-- 2. POLICIES

create table if not exists public.policies (
    policy_id text primary key,

    policy_name text not null,

    description text,

    created_at timestamptz not null default now()
);

-- 3. POLICY VERSIONS


create table if not exists public.policy_versions (
    policy_version_id bigint generated always as identity primary key,

    policy_id text not null
        references public.policies(policy_id)
        on update cascade
        on delete restrict,

    version text not null,

    section text,

    policy_text text not null,

    created_at timestamptz not null default now(),

    unique (policy_id, version, section)
);


-- 4. FINDINGS

create table if not exists public.findings (
    finding_id text primary key,

    audit_run_id text not null
        references public.audit_runs(audit_run_id)
        on update cascade
        on delete restrict,

    control_id text not null,

    customer_id text,

    severity text not null
        check (
            severity in (
                'LOW',
                'MEDIUM',
                'HIGH',
                'CRITICAL'
            )
        ),

    assessment_status text not null
        check (
            assessment_status in (
                'PASS',
                'FAIL',
                'NOT_APPLICABLE'
            )
        ),

    finding_status text not null
        check (
            finding_status in (
                'REVIEW',
                'CONFIRMED',
                'REJECTED',
                'RESOLVED'
            )
        ),

    expected text not null,

    actual text not null,

    evidence jsonb not null default '{}'::jsonb,

    policy_references jsonb not null default '[]'::jsonb,

    reviewed_by text,

    review_timestamp timestamptz,

    reviewer_notes text,

    ai_explanation text,

    ai_recommendation text,

    created_at timestamptz not null default now(),

    updated_at timestamptz not null default now(),

    -- A finding must belong to an audit run.
    constraint findings_audit_run_fk
        foreign key (audit_run_id)
        references public.audit_runs(audit_run_id)
        on update cascade
        on delete restrict,

    -- Finding identity used by the current evaluation strategy.
    constraint findings_identity_unique
        unique (audit_run_id, control_id, customer_id)
);


-- 5. DETERMINISTIC FINDING EXPLANATIONS

create table if not exists public.finding_explanations (
    explanation_id bigint generated always as identity primary key,

    finding_id text not null
        references public.findings(finding_id)
        on update cascade
        on delete cascade,

    audit_run_id text not null
        references public.audit_runs(audit_run_id)
        on update cascade
        on delete restrict,

    control_id text not null,

    customer_id text,

    severity text not null
        check (
            severity in (
                'LOW',
                'MEDIUM',
                'HIGH',
                'CRITICAL'
            )
        ),

    assessment_status text not null
        check (
            assessment_status in (
                'PASS',
                'FAIL',
                'NOT_APPLICABLE'
            )
        ),

    finding_status text not null
        check (
            finding_status in (
                'REVIEW',
                'CONFIRMED',
                'REJECTED',
                'RESOLVED'
            )
        ),

    summary text not null,

    expected_condition text not null,

    observed_condition text not null,

    evidence jsonb not null default '{}'::jsonb,

    policy_references text[] not null default '{}',

    review_action text,

    created_at timestamptz not null default now(),

    -- One deterministic explanation per finding.
    unique (finding_id)
);

-- 6. HUMAN FINDING REVIEWS

create table if not exists public.finding_reviews (
    review_id bigint generated always as identity primary key,

    finding_id text not null
        references public.findings(finding_id)
        on update cascade
        on delete cascade,

    audit_run_id text not null
        references public.audit_runs(audit_run_id)
        on update cascade
        on delete restrict,

    previous_status text not null
        check (
            previous_status in (
                'REVIEW',
                'CONFIRMED',
                'REJECTED',
                'RESOLVED'
            )
        ),

    new_status text not null
        check (
            new_status in (
                'CONFIRMED',
                'REJECTED',
                'RESOLVED'
            )
        ),

    reviewed_by text not null,

    reviewer_notes text,

    reviewed_at timestamptz not null default now()
);


-- 7. AI OUTPUTS

create table if not exists public.ai_outputs (
    ai_output_id bigint generated always as identity primary key,

    finding_id text not null
        references public.findings(finding_id)
        on update cascade
        on delete cascade,

    audit_run_id text not null
        references public.audit_runs(audit_run_id)
        on update cascade
        on delete restrict,

    ai_explanation text,

    ai_recommendation text,

    model_name text,

    prompt_version text,

    retrieved_policy_context jsonb not null default '[]'::jsonb,

    created_at timestamptz not null default now()
);

-- 8. INDEXES

create index if not exists idx_findings_audit_run
    on public.findings(audit_run_id);

create index if not exists idx_findings_control
    on public.findings(control_id);

create index if not exists idx_findings_customer
    on public.findings(customer_id);

create index if not exists idx_findings_status
    on public.findings(finding_status);

create index if not exists idx_findings_severity
    on public.findings(severity);

create index if not exists idx_finding_reviews_finding
    on public.finding_reviews(finding_id);

create index if not exists idx_finding_explanations_finding
    on public.finding_explanations(finding_id);

create index if not exists idx_ai_outputs_finding
    on public.ai_outputs(finding_id);

create index if not exists idx_policy_versions_policy
    on public.policy_versions(policy_id);

-- 9. UPDATED_AT TRIGGER

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;


drop trigger if exists trg_findings_updated_at
on public.findings;

create trigger trg_findings_updated_at
before update on public.findings
for each row
execute function public.set_updated_at();


-- 10. ROW LEVEL SECURITY

alter table public.audit_runs enable row level security;
alter table public.findings enable row level security;
alter table public.finding_explanations enable row level security;
alter table public.finding_reviews enable row level security;
alter table public.policies enable row level security;
alter table public.policy_versions enable row level security;
alter table public.ai_outputs enable row level security;