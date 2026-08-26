
-- AI-Powered Pre-Audit & Reconciliation Platform
-- Migration: 004_rls_read_policies
--
-- Scope:
--   Adds the RLS policies that were deferred in 002_schema_hardening
--   ("RLS policies: service_role bypasses RLS regardless of
--   policies, so none are needed until a non-service-role caller
--   ... needs direct Supabase access. Decide the access model
--   then.").
--
--   That caller is now anticipated: backend/main.py's read-only
--   (GET) endpoints are the intended eventual home for a
--   lower-privilege key (anon or authenticated), once the API layer
--   is switched over from SUPABASE_SERVICE_ROLE_KEY. Until that
--   switch happens, this migration changes NOTHING about current
--   backend behavior -- service_role bypasses RLS entirely
--   regardless of what policies exist. This migration only defines,
--   in advance, what a non-service-role caller would be allowed to
--   see, so the app-layer switch (when it happens) is a client-key
--   change, not a fresh RLS design exercise done under time
--   pressure.
--
-- Design principle:
--   READ-ONLY. anon and authenticated may SELECT from the tables
--   that the findings-list, finding-detail, and dashboard-summary
--   endpoints expose. No INSERT/UPDATE/DELETE policy is granted to
--   anon or authenticated anywhere in this migration -- every write
--   in this platform (audit execution, human review decisions, AI
--   explanation generation) goes through the backend's own
--   API-key-gated endpoints (see backend/auth.py), which use
--   service_role and therefore neither need nor want an RLS write
--   policy. A write policy for anon/authenticated would create a
--   second, unaudited path to modify audit data that bypasses the
--   application's own API-key check entirely -- deliberately not
--   doing that here.
--
-- Deliberately excluded from anon/authenticated SELECT:
--   finding_reviews -- contains reviewer identity (reviewed_by) and
--   free-text reviewer notes, which is internal audit-review
--   correspondence, not findings-list content that a broader
--   audience should see by default. RLS stays enabled on this table
--   with no anon/authenticated policy, so it remains
--   service_role-only (i.e. denied for every other role) until a
--   deliberate, separate decision is made to expose review history
--   more broadly.
--
-- Follow-up (not done here):
--   Switching backend/database.py to actually use the anon key for
--   the GET endpoints above -- that's an application-layer change,
--   tracked separately from this schema migration.


-- 1. AUDIT_RUNS -- read-only

drop policy if exists "Allow read access to audit_runs" on public.audit_runs;

create policy "Allow read access to audit_runs"
on public.audit_runs
for select
to anon, authenticated
using (true);


-- 2. FINDINGS -- read-only

drop policy if exists "Allow read access to findings" on public.findings;

create policy "Allow read access to findings"
on public.findings
for select
to anon, authenticated
using (true);


-- 3. FINDING_EXPLANATIONS -- read-only
--    (deterministic Stage 2 explanation -- never the AI/LLM output)

drop policy if exists "Allow read access to finding_explanations" on public.finding_explanations;

create policy "Allow read access to finding_explanations"
on public.finding_explanations
for select
to anon, authenticated
using (true);


-- 4. POLICIES -- read-only

drop policy if exists "Allow read access to policies" on public.policies;

create policy "Allow read access to policies"
on public.policies
for select
to anon, authenticated
using (true);


-- 5. POLICY_VERSIONS -- read-only

drop policy if exists "Allow read access to policy_versions" on public.policy_versions;

create policy "Allow read access to policy_versions"
on public.policy_versions
for select
to anon, authenticated
using (true);


-- 6. AI_OUTPUTS -- read-only
--    (LLM-generated explanation/recommendation, once Stage 3 exists)

drop policy if exists "Allow read access to ai_outputs" on public.ai_outputs;

create policy "Allow read access to ai_outputs"
on public.ai_outputs
for select
to anon, authenticated
using (true);


-- 7. AUDIT_EVALUATIONS -- read-only
--    (TP / FP / FN, precision / recall / F1 per audit run)

drop policy if exists "Allow read access to audit_evaluations" on public.audit_evaluations;

create policy "Allow read access to audit_evaluations"
on public.audit_evaluations
for select
to anon, authenticated
using (true);


-- 8. FINDING_REVIEWS -- intentionally NO policy added here.
--
--    RLS is already enabled on this table (see 001_initial_audit_
--    schema). With RLS enabled and zero matching policies, anon and
--    authenticated are denied by default -- which is the desired
--    state until reviewer identity / notes are deliberately decided
--    to be exposed. service_role continues to bypass RLS as always,
--    so the backend's own review endpoints are unaffected.
