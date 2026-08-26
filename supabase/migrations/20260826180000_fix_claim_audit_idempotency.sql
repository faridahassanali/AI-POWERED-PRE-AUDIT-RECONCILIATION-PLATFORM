-- Atomically claim an idempotency key for an audit run.
--
-- First request:
--   inserts the audit_run and returns claimed=true.
--
-- Duplicate request:
--   does not insert a new audit_run and returns
--   claimed=false together with the original audit_run_id.

CREATE OR REPLACE FUNCTION public.claim_audit_idempotency(
    p_audit_run_id text,
    p_idempotency_key text
)
RETURNS TABLE (
    claimed boolean,
    audit_run_id text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF p_audit_run_id IS NULL
       OR trim(p_audit_run_id) = '' THEN
        RAISE EXCEPTION
            'audit_run_id must not be empty';
    END IF;

    IF p_idempotency_key IS NULL
       OR trim(p_idempotency_key) = '' THEN
        RAISE EXCEPTION
            'idempotency_key must not be empty';
    END IF;

    -- First request: atomically insert the audit run.
    INSERT INTO public.audit_runs (
        audit_run_id,
        idempotency_key,
        status,
        started_at
    )
    VALUES (
        p_audit_run_id,
        p_idempotency_key,
        'RUNNING',
        now()
    )
    ON CONFLICT (idempotency_key)
    DO NOTHING;

    -- Return the actual state associated with the key.
    --
    -- If this request inserted the row, the ID will be p_audit_run_id.
    -- If another request already claimed the key, this returns
    -- the existing audit_run_id.
    RETURN QUERY
    SELECT
        (ar.audit_run_id = p_audit_run_id) AS claimed,
        ar.audit_run_id
    FROM public.audit_runs AS ar
    WHERE ar.idempotency_key = p_idempotency_key
    LIMIT 1;
END;
$$;

REVOKE EXECUTE
ON FUNCTION public.claim_audit_idempotency(text, text)
FROM PUBLIC;

GRANT EXECUTE
ON FUNCTION public.claim_audit_idempotency(text, text)
TO service_role;