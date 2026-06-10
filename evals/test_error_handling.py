"""
test_error_handling.py — negative-path / failure-mode evals.

Where test_quality.py grades the agent's outputs on clean inputs, this file
grades the agent's BEHAVIOR on bad or unsafe inputs: does it fail *correctly*?

Every case here short-circuits before any LLM call — input validation at
/api/submit, or the NODE 0 PII guardrail that halts on sensitive PII — so the
whole file runs deterministically, with no judge model and no OpenAI billing.
That makes it safe to run on every push (no fixture refresh required):

    cd <project root>
    pytest evals/test_error_handling.py

The guard-level "known gap" tests document detection holes (undashed SSN, PII
in the job description) as xfail. When the guard is hardened to close a gap,
its xfail flips to an unexpected pass — a signal to promote it to a hard assert.
"""

import pytest
from dotenv import load_dotenv
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PKG_ROOT / ".env")

from evals.dataset import ADVERSARIAL_EXAMPLES
from evals.api_client import run_via_api
from guardrails.pii_guard import scan_pii


# ── API-level error handling (deterministic, no judge) ───────────────────────────

@pytest.mark.parametrize("ex", ADVERSARIAL_EXAMPLES, ids=lambda e: e.name)
def test_adversarial_input_is_handled(ex):
    result = run_via_api(ex.job_description, ex.resume_text)

    # 1. Correct failure mode — not a crash, hang, or silent success.
    assert result["outcome"] == ex.expect, (
        f"{ex.name}: expected outcome '{ex.expect}', got '{result['outcome']}' "
        f"(message: {result['message']!r})"
    )

    # 2. The message explains WHY, so a user/operator can act on it.
    if ex.expect_message_contains:
        assert ex.expect_message_contains.lower() in result["message"].lower(), (
            f"{ex.name}: message {result['message']!r} is missing "
            f"{ex.expect_message_contains!r}"
        )

    # 3. Sensitive values never leak back into a surfaced message/event.
    if ex.must_not_leak:
        assert ex.must_not_leak not in result["message"], (
            f"{ex.name}: sensitive value {ex.must_not_leak!r} leaked into the "
            f"error message — it must never be echoed back."
        )


def test_sensitive_pii_blocks_before_any_llm_call():
    """
    The SSN case must reach 'error' via the PII guard, not via a downstream LLM
    failure — i.e. no output payload is ever produced for a blocked resume.
    """
    ex = next(e for e in ADVERSARIAL_EXAMPLES if e.name == "ssn_in_resume")
    result = run_via_api(ex.job_description, ex.resume_text)
    assert result["outcome"] == "error"
    assert result["outputs"] is None


# ── Guard-level characterization: confirmed-good detection ───────────────────────

def test_dashed_ssn_in_resume_is_detected_sensitive():
    report, _ = scan_pii("Priya Sharma\nSSN: 123-45-6789\n")
    assert report.has_sensitive and "SSN" in report.sensitive_types


# ── Guard-level KNOWN GAPS (xfail until the guard is hardened) ────────────────────
#
# These assert the DESIRED behavior. They currently fail because the guard does
# not yet cover these shapes. When someone tightens the regex / scans the JD,
# the xfail becomes an unexpected pass — promote it to a plain assert then.

@pytest.mark.xfail(reason="known gap: SSN without dashes is not detected", strict=False)
def test_undashed_ssn_should_be_detected():
    report, _ = scan_pii("SSN: 123456789")
    assert report.has_sensitive, "undashed SSN should be treated as sensitive"


@pytest.mark.xfail(reason="known gap: PII guard only scans the resume, not the JD", strict=False)
def test_ssn_in_job_description_should_block():
    # scan_pii is input-agnostic and DOES flag an SSN anywhere...
    jd = "Backend role. Reference candidate SSN 123-45-6789 for verification."
    assert scan_pii(jd)[0].has_sensitive
    # ...but the NODE 0 guardrail only inspects resume_text, so a JD-borne SSN
    # sails through. We assert the DESIRED behavior — the node should block on a
    # sensitive JD too — by calling the node directly (no LLM, no billing).
    from agents.nodes import pii_guardrail
    from guardrails.pii_guard import PIIBlockedError
    with pytest.raises(PIIBlockedError):
        pii_guardrail({
            "resume_text": "Priya Sharma\nBackend engineer, 5 years Python.\n",
            "job_description": jd,
        })
