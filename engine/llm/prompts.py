"""
LLM Layer -- Prompt Design.

ONE prompt design, shared by every provider (Groq, Gemini, ...), so
switching providers on failover can never change what the model is
asked to do -- only which model answers.

=====================================================================
FOR WHOEVER BUILDS TASK B (engine/ai_input.py):
=====================================================================

This prompt consumes ai_input exactly as build_ai_input() produces
it. The fields it reads are:

    finding_id            str
    audit_run_id          str
    control_id            str
    customer_id           str
    severity              str
    assessment_status     str
    expected               str  (or "expected_condition")
    actual                  str  (or "observed_condition")
    evidence                dict
    policy_context          list[dict] -- each item MUST have at
                             least: policy_id, version, section,
                             content (the resolved policy chunk text)
    reviewed_by             str
    review_timestamp        str
    reviewer_notes           str | None

If a field name changes on your end (e.g. "expected" vs
"expected_condition"), update _get() calls below rather than
changing the prompt text itself -- keep the wording stable so
provider outputs stay comparable across iterations.

policy_context is the ONLY source of policy text the model is
allowed to cite. If Task A/the bridge ever changes what a
policy_context chunk looks like, the "content" key is the one
field this prompt absolutely depends on -- everything else
degrades gracefully if missing.

OPTIONAL RETRY FIELD:

    _retry_reminder    str | None

This is NOT part of the ai_input contract build_ai_input() produces
-- it is attached (and later discarded) by
engine.ai_explanation_pipeline ONLY when a prior attempt for this
same finding was rejected by the hallucination tripwire
(engine.llm.hallucination_tripwire). When present, build_user_prompt()
appends it as a clearly-marked correction block at the end of the
user message, quoting the exact tripwire complaints, so the retried
call is pointed directly at what it got wrong instead of blindly
trying again with the same prompt.
"""

import json
from typing import Any


# =====================================================================
# SYSTEM PROMPT
# =====================================================================
#
# This is sent once per call, identically to every provider. It is
# never shown to the human reviewer -- it's the model's operating
# instructions, not part of the explanation output.

SYSTEM_PROMPT = """\
You are a compliance explanation assistant for a bank's internal \
pre-audit reconciliation platform.

You do not investigate, decide, or re-evaluate audit findings. A \
deterministic rule engine has already determined the finding's \
status, severity, expected condition, and actual condition. A human \
compliance reviewer has already confirmed this finding is valid. \
Your only job is to explain, in clear professional language, WHY \
this finding violates the applicable policy and WHAT the reviewer \
should consider doing about it.

Hard rules, in priority order:

1. GROUNDING. You may only reference policy text that appears in the \
   "policy_context" you are given below. Never cite a policy_id, \
   version, or section that is not present there. Never state a \
   policy requirement from general knowledge, memory, or assumption \
   -- if policy_context does not clearly support a claim, do not \
   make that claim.

2. NO RE-JUDGING. Never contradict, soften, or second-guess the \
   finding's severity, assessment_status, or the reviewer's \
   decision to confirm it. Treat those as settled facts.

3. NO FABRICATION. Only use facts present in the finding's evidence, \
   expected/actual conditions, and policy_context. Do not invent \
   customer details, dates, or amounts not given to you.

4. OUTPUT FORMAT. Respond with ONLY a single JSON object, no prose \
   before or after it, matching exactly this shape:

   {
     "explanation": "<2-4 sentences: what was specifically found for \
this finding, then the policy requirement it violates, then why \
that is a violation>",
     "recommendation": "<1-3 sentences: a concrete, actionable next \
step for the reviewer, grounded in the same policy>",
     "cited_policy_ids": ["<policy_id values you actually relied on, \
must be a subset of the policy_id values in policy_context>"]
   }

5. LANGUAGE AND TONE. Write for a bank compliance reviewer who is \
   not a software engineer. Plain, professional, precise. English \
   unless the finding's evidence is primarily Arabic-script content \
   (e.g. ARABIC_NAME_001), in which case you may note the Arabic \
   text as-is but still explain in English.

6. IF POLICY_CONTEXT IS INSUFFICIENT. If the given policy_context \
   does not clearly support explaining this finding, say so plainly \
   in "explanation" instead of guessing, and return an empty \
   "cited_policy_ids" list.

7. LEAD WITH THE SPECIFIC FACT. Open "explanation" with what is \
   specifically true about THIS finding (the customer's actual \
   value, the specific mismatch, the specific missing field) -- not \
   with a restatement of the policy requirement. State the policy \
   requirement second, to support the fact you just gave, not as the \
   opening sentence. Two findings of the same control_id should read \
   as clearly distinct because they lead with different specifics, \
   even though they cite the same policy.

8. PRESERVE THE POLICY'S ACTUAL MODAL STRENGTH. Policy text \
   distinguishes "must" / "must not" (mandatory) from "may" / \
   "should" (permissive/discretionary) -- carry that distinction \
   through exactly as written. Do not soften a "must not" into "may \
   not" or "should not" when paraphrasing, and do not invent a "must" \
   where the policy only says "may". If a status being CLEAR is what \
   permits a wallet to proceed ("may proceed"), the violation you are \
   explaining is usually the mandatory prohibition elsewhere in the \
   policy (e.g. "must not be treated as clear", "must not be \
   activated") -- cite THAT mandatory language for the violation \
   itself, not the permissive language for the allowed path.

9. ADDRESS CONFLICTING REVIEWER NOTES EXPLICITLY. If reviewer_notes \
   is present and states or implies a reason the finding might be \
   acceptable (e.g. citing a mitigating field, a business \
   justification, or a reason the violation shouldn't count), and \
   that reason does NOT actually satisfy the mandatory policy \
   requirement, do not silently ignore the note and do not silently \
   contradict it either -- explicitly name what the note claims, then \
   explain in one sentence why it does not change the outcome (e.g. \
   "The reviewer noted the screening reference and evidence are on \
   file; however, the policy requires a CLEAR result specifically, \
   and evidence being present does not satisfy that -- HIGH_RISK must \
   still not be treated as clear."). This still never re-judges the \
   finding itself (rule 2) -- it only makes the reasoning visible so \
   the next reader isn't left wondering why the explanation reads as \
   if it ignored what the reviewer wrote. If reviewer_notes is absent, \
   or does not conflict with the violation, skip this entirely -- do \
   not manufacture a rebuttal to a note that isn't there.

10. NEVER CONFLATE SEVERITY WITH A STATUS VALUE. "severity" (LOW / \
    MEDIUM / HIGH / CRITICAL) tells you how urgent this finding is -- \
    it is NOT a screening/account/customer status and must never be \
    restated as one. In particular, a finding with severity "HIGH" \
    does NOT mean any field's value is "HIGH_RISK" -- those are two \
    unrelated words. Only ever state that a specific field (e.g. a \
    screening result, an account status) equals a given value when \
    that exact value is literally present in this finding's own \
    "evidence", "expected", or "actual" fields below. If you want to \
    convey urgency, say "high severity" in plain prose -- never as a \
    quoted or status-like token that could be mistaken for an \
    evidence value.
"""


# =====================================================================
# USER PROMPT BUILDER
# =====================================================================

def _get(ai_input: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """
    Try several possible field names in order, so small naming
    differences on the ai_input side (e.g. "expected" vs
    "expected_condition") don't require touching prompt text.
    """

    for key in keys:
        if key in ai_input and ai_input[key] not in (None, ""):
            return ai_input[key]

    return default


def _format_policy_context(
    policy_context: list[dict[str, Any]],
) -> str:
    """
    Render the resolved policy chunks as the ONLY policy text the
    model will see. Each chunk is clearly delimited and labeled with
    its policy_id/version/section, since cited_policy_ids must match
    these exactly.
    """

    if not policy_context:
        return "(no policy_context was resolved for this finding)"

    blocks = []

    for i, chunk in enumerate(policy_context, start=1):

        policy_id = chunk.get("policy_id", "UNKNOWN")
        version = chunk.get("version", "")
        section = chunk.get("section", "")
        content = chunk.get("content", "")

        blocks.append(
            f"[{i}] policy_id={policy_id} version={version} "
            f"section=\"{section}\"\n{content}"
        )

    return "\n\n".join(blocks)


def build_user_prompt(ai_input: dict[str, Any]) -> str:
    """
    Build the per-finding user message. Deterministic string
    building only -- no model calls here.
    """

    control_id = _get(ai_input, "control_id")
    customer_id = _get(ai_input, "customer_id")
    severity = _get(ai_input, "severity")
    assessment_status = _get(ai_input, "assessment_status")
    expected = _get(ai_input, "expected", "expected_condition")
    actual = _get(ai_input, "actual", "observed_condition")
    evidence = _get(ai_input, "evidence", default={})
    reviewer_notes = _get(ai_input, "reviewer_notes", default=None)
    policy_context = _get(ai_input, "policy_context", default=[])
    retry_reminder = _get(ai_input, "_retry_reminder", default=None)

    evidence_json = json.dumps(
        evidence,
        ensure_ascii=False,
        indent=2,
    )

    lines = [
        f"control_id: {control_id}",
        f"customer_id: {customer_id}",
        f"severity: {severity}",
        f"assessment_status: {assessment_status}",
        f"expected: {expected}",
        f"actual: {actual}",
        f"evidence: {evidence_json}",
    ]

    if reviewer_notes:
        lines.append(f"reviewer_notes: {reviewer_notes}")

    lines.append("")
    lines.append("policy_context (the ONLY policy text you may cite):")
    lines.append(_format_policy_context(policy_context))

    if retry_reminder:
        lines.append("")
        lines.append("=== CORRECTION REQUIRED (previous attempt rejected) ===")
        lines.append(str(retry_reminder))
        lines.append(
            "Rewrite the explanation/recommendation from scratch. Do not "
            "repeat the rejected claim(s) above in any form."
        )

    return "\n".join(lines)


# =====================================================================
# JSON SCHEMA (used by providers that support structured output)
# =====================================================================

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "recommendation": {"type": "string"},
        "cited_policy_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "explanation",
        "recommendation",
        "cited_policy_ids",
    ],
}