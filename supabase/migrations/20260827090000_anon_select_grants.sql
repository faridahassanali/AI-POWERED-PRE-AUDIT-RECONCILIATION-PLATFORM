-- AI-Powered Pre-Audit & Reconciliation Platform
-- Migration: 005_anon_select_grants
--
-- Scope:
--   RLS policies for anon/authenticated were added in
--   20260826140000_rls_read_policies.sql, but RLS policies alone are
--   NOT sufficient in Postgres -- a role also needs the underlying
--   table-level GRANT before RLS policies are evaluated.
--
--   20260824110000_service_role_grants.sql only granted privileges
--   to service_role, never to anon/authenticated. That gap is what
--   produces:
--
--       permission denied for table findings (42501)
--       hint: Grant the required privileges to the current role
--       with: GRANT SELECT ON public.findings TO anon;
--
--   This migration adds SELECT-only grants for anon/authenticated,
--   matching the read-only access model established by the RLS
--   policies in 20260826140000_rls_read_policies.sql.
--
--   No INSERT/UPDATE/DELETE grant is added for anon/authenticated.
--   Every write in this platform goes through the backend's
--   API-key-gated endpoints using service_role.
--
--   finding_reviews is included with a SELECT grant because the
--   database preflight verifier requires anon to have the underlying
--   table-level SELECT privilege on every expected table.
--
--   The SELECT grant does NOT bypass RLS. RLS policies continue to
--   control which rows the role is actually allowed to read.
--
-- Also grants USAGE on the public schema itself -- required for a
-- role to resolve objects in that schema. Supabase's default project
-- setup normally provisions this for anon/authenticated already,
-- but it is included defensively so this migration is self-contained
-- and does not depend on an assumption about the project's initial state.


GRANT USAGE
ON SCHEMA public
TO anon, authenticated;


GRANT SELECT
ON TABLE public.audit_runs
TO anon, authenticated;


GRANT SELECT
ON TABLE public.findings
TO anon, authenticated;


GRANT SELECT
ON TABLE public.finding_explanations
TO anon, authenticated;


GRANT SELECT
ON TABLE public.policies
TO anon, authenticated;


GRANT SELECT
ON TABLE public.policy_versions
TO anon, authenticated;


GRANT SELECT
ON TABLE public.ai_outputs
TO anon, authenticated;


GRANT SELECT
ON TABLE public.audit_evaluations
TO anon, authenticated;


-- finding_reviews is included because it is one of the expected
-- database tables checked by scripts/verify_database.py.
-- RLS remains responsible for restricting which rows anon can read.

GRANT SELECT
ON TABLE public.finding_reviews
TO anon;