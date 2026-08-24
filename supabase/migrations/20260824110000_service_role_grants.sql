-- Service-role grants for application persistence.
-- RLS remains enabled; service_role bypasses RLS,
-- but explicit table privileges are required.

GRANT SELECT, INSERT, UPDATE
ON TABLE public.policies
TO service_role;

GRANT SELECT, INSERT, UPDATE
ON TABLE public.policy_versions
TO service_role;

GRANT SELECT, INSERT, UPDATE
ON TABLE public.audit_runs
TO service_role;

GRANT SELECT, INSERT, UPDATE
ON TABLE public.findings
TO service_role;

GRANT SELECT, INSERT, UPDATE
ON TABLE public.finding_reviews
TO service_role;

GRANT SELECT, INSERT, UPDATE
ON TABLE public.finding_explanations
TO service_role;

GRANT SELECT, INSERT, UPDATE
ON TABLE public.ai_outputs
TO service_role;

GRANT SELECT, INSERT, UPDATE
ON TABLE public.audit_evaluations
TO service_role;