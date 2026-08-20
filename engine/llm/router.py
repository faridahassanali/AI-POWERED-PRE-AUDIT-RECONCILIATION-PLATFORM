"""
LLM Layer -- Router.

Single entry point for the rest of the pipeline: explain(ai_input).

Everything else (engine/finding_explainer.py, the Stage 2 caller,
etc.) should import ONLY this function -- never GroqProvider or
GeminiProvider directly. That's what makes the failover invisible to
the rest of the system: the explanation shape coming out of
explain() is identical no matter which provider actually answered.

Failover policy
----------------

    LLMTransientError  -> retry primary once (short backoff), then
                           fail over to the fallback provider.
    LLMOutputError      -> retry the SAME provider once (a malformed
                           JSON response is usually a one-off), then
                           fail over if it happens again.
    LLMConfigError       -> raised immediately, no retry, no
                           failover. A bad/missing API key is a
                           standing problem -- silently routing
                           every call to the fallback would hide a
                           broken primary indefinitely, which is
                           worse than a loud failure in a compliance
                           system.

If both providers ultimately fail, explain() raises LLMError with
both underlying errors attached, so the caller (Stage 2) can log a
single clear failure instead of swallowing the finding silently.
"""

import logging
import time
from typing import Any

from engine.llm.base import (
    LLMConfigError,
    LLMError,
    LLMExplanation,
    LLMOutputError,
    LLMProvider,
    LLMTransientError,
)
from engine.llm.gemini_provider import GeminiProvider
from engine.llm.groq_provider import GroqProvider


logger = logging.getLogger(__name__)

RETRY_BACKOFF_SECONDS = 0.5


class LLMAllProvidersFailedError(LLMError):
    """
    Raised when both the primary and the fallback provider failed.

    Carries the individual errors so the caller can log or persist
    the real cause instead of a generic message.
    """

    def __init__(
        self,
        primary_error: Exception,
        fallback_error: Exception,
    ) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error

        super().__init__(
            "Both LLM providers failed. "
            f"primary: {type(primary_error).__name__}: {primary_error} | "
            f"fallback: {type(fallback_error).__name__}: {fallback_error}"
        )


def _attempt(
    provider: LLMProvider,
    ai_input: dict[str, Any],
    retry_on_output_error: bool = True,
) -> LLMExplanation:
    """
    Call one provider, with a single same-provider retry ONLY for
    LLMOutputError (malformed JSON is often a one-off formatting
    slip). LLMTransientError and LLMConfigError are not retried
    here -- the router-level retry/backoff for LLMTransientError
    happens in explain(), and LLMConfigError never retries.
    """

    try:
        return provider.generate(ai_input)

    except LLMOutputError:

        if not retry_on_output_error:
            raise

        logger.warning(
            "%s returned a malformed response; retrying once.",
            provider.name,
        )

        return provider.generate(ai_input)


def explain(
    ai_input: dict[str, Any],
    primary: LLMProvider | None = None,
    fallback: LLMProvider | None = None,
) -> dict[str, Any]:
    """
    Generate a grounded explanation for a CONFIRMED, gated finding.

    ai_input must already be the output of
    engine.ai_input.build_ai_input() -- this function does not
    re-validate the confirmed-only gate or the non-empty
    policy_context requirement; that's build_ai_input()'s job.

    Returns a plain dict (LLMExplanation.to_dict()) so callers don't
    need to import the dataclass.

    primary/fallback are injectable for testing; production callers
    can omit both and get GroqProvider() / GeminiProvider().
    """

    primary = primary if primary is not None else GroqProvider()
    fallback = fallback if fallback is not None else GeminiProvider()

    # -----------------------------------------------------------
    # 1. TRY PRIMARY (Groq)
    # -----------------------------------------------------------

    try:
        result = _attempt(primary, ai_input)
        return result.to_dict()

    except LLMConfigError:
        # Not retryable, not fail-over-able -- see module docstring.
        raise

    except (LLMTransientError, LLMOutputError) as primary_error:

        logger.warning(
            "Primary provider (%s) failed: %s. Retrying once before "
            "failing over.",
            primary.name,
            primary_error,
        )

        time.sleep(RETRY_BACKOFF_SECONDS)

        try:
            result = _attempt(primary, ai_input, retry_on_output_error=False)
            return result.to_dict()

        except LLMConfigError:
            raise

        except (LLMTransientError, LLMOutputError) as primary_retry_error:

            logger.warning(
                "Primary provider (%s) failed again: %s. Failing "
                "over to %s.",
                primary.name,
                primary_retry_error,
                fallback.name,
            )

            # -----------------------------------------------------------
            # 2. FAIL OVER (Gemini)
            # -----------------------------------------------------------

            try:
                result = _attempt(fallback, ai_input)

                logger.info(
                    "Explanation for %s generated by fallback provider "
                    "(%s) after primary (%s) failed.",
                    ai_input.get("finding_id"),
                    fallback.name,
                    primary.name,
                )

                return result.to_dict()

            except LLMConfigError:
                raise

            except (LLMTransientError, LLMOutputError) as fallback_error:

                raise LLMAllProvidersFailedError(
                    primary_error=primary_retry_error,
                    fallback_error=fallback_error,
                ) from fallback_error
