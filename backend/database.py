import os

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the .env file."
    )

# service_role -- bypasses RLS entirely. Used ONLY for write paths
# (POST/PATCH endpoints, already gated behind backend.auth.verify_api_key)
# and for anything that must see every row regardless of policy
# (e.g. create_finding_review / get_finding_reviews).
supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)

# anon -- subject to RLS (see supabase/migrations/
# 20260826140000_rls_read_policies.sql). Used for the read-only (GET)
# endpoints that don't need elevated access, so a bug or future
# endpoint added to this client can never see or touch more than the
# RLS policies explicitly allow -- unlike the service_role client,
# which bypasses RLS by design and must never be used for a path
# whose exposure should be controlled by policy.
#
# Falls back to None if SUPABASE_ANON_KEY isn't set, so existing
# deployments that haven't added the key yet don't crash on import;
# callers that need it must handle the None case (see
# backend/main.py's _read_client() helper).
supabase_anon: Client | None = (
    create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    if SUPABASE_ANON_KEY
    else None
)
