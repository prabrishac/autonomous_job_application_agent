"""
dataset.py — Golden evaluation inputs for the deepeval suite.

Each example is exactly the payload a user submits to POST /api/submit:
a job description + a resume.  The resumes intentionally include real-looking
contact details so the PII guardrail has something to redact/restore, and they
are deliberately *narrower* than the JD so the "no fabrication" metric has a
chance to catch the agent inventing experience the candidate never had.

Keep this list small — every example triggers a full agent run (web search +
~6 LLM calls), so the suite is billed per example.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    name: str            # short slug, also the fixture filename
    job_description: str
    resume_text: str
    # Expected outcome of submitting this (JD, resume) through the API.
    #   "success"     → run reaches the human-review interrupt with outputs
    #   "error"       → run starts but the background agent halts (status=error)
    #   "reject_400"  → POST /api/submit rejects the payload outright (HTTP 400)
    expect: str = "success"
    # Substring the error/rejection message MUST contain (case-insensitive).
    expect_message_contains: str = ""
    # A raw value that must NOT appear in any surfaced message/event (leak guard).
    must_not_leak: str = ""


GOLDEN_EXAMPLES: list[Example] = [
    Example(
        name="backend_engineer_stripe",
        job_description=(
            "Software Engineer, Backend — Stripe\n\n"
            "Stripe builds economic infrastructure for the internet. We are hiring a "
            "backend engineer to design and operate payment APIs at scale.\n\n"
            "What you'll do:\n"
            "- Build and maintain high-throughput services in Python and Go.\n"
            "- Design REST and gRPC APIs used by millions of businesses.\n"
            "- Improve reliability of distributed systems handling money movement.\n"
            "- Partner with product teams to ship features end to end.\n\n"
            "Requirements:\n"
            "- 4+ years building backend services in production.\n"
            "- Strong with Python; experience with PostgreSQL and Kafka.\n"
            "- Experience with distributed systems and API design.\n"
            "- Nice to have: payments domain knowledge, Go, Kubernetes.\n"
        ),
        resume_text=(
            "Priya Sharma\n"
            "priya.sharma@example.com | (415) 555-0188 | San Francisco, CA\n\n"
            "Summary\n"
            "Backend engineer with 5 years building Python services for high-traffic apps.\n\n"
            "Experience\n"
            "Senior Software Engineer, Acme Logistics (2021–present)\n"
            "- Built REST APIs in Python/FastAPI serving 20k requests/sec.\n"
            "- Designed PostgreSQL schemas and optimized slow queries by 60%.\n"
            "- Introduced Kafka for order-event streaming across 4 services.\n\n"
            "Software Engineer, Bytewave (2019–2021)\n"
            "- Maintained a Django monolith and split out two microservices.\n"
            "- Added Redis caching that cut p95 latency from 800ms to 220ms.\n\n"
            "Skills: Python, FastAPI, Django, PostgreSQL, Kafka, Redis, Docker, AWS\n"
        ),
    ),
    Example(
        name="data_scientist_airbnb",
        job_description=(
            "Data Scientist, Marketplace — Airbnb\n\n"
            "Airbnb connects people to unique travel experiences. Our Marketplace "
            "team is hiring a data scientist to drive pricing and matching decisions.\n\n"
            "What you'll do:\n"
            "- Design and analyze A/B experiments for pricing and search ranking.\n"
            "- Build predictive models (Python, SQL) that inform product strategy.\n"
            "- Communicate findings to product and engineering leaders.\n\n"
            "Requirements:\n"
            "- 3+ years in applied data science or analytics.\n"
            "- Strong SQL and Python (pandas, scikit-learn).\n"
            "- Experience with experimentation and causal inference.\n"
            "- Nice to have: marketplace/pricing experience, Spark.\n"
        ),
        resume_text=(
            "Daniel Okoro\n"
            "daniel.okoro@example.com | +1 (646) 555-0143 | Brooklyn, NY\n\n"
            "Summary\n"
            "Data scientist with 4 years turning product data into decisions.\n\n"
            "Experience\n"
            "Data Scientist, Lumen Retail (2021–present)\n"
            "- Ran 30+ A/B tests on checkout flow; lifted conversion 7%.\n"
            "- Built churn model in scikit-learn used by the CRM team.\n"
            "- Wrote SQL pipelines over 200M-row event tables.\n\n"
            "Analytics Associate, Northwind (2020–2021)\n"
            "- Built weekly KPI dashboards in Python and Looker.\n\n"
            "Skills: Python, pandas, scikit-learn, SQL, A/B testing, statistics\n"
        ),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial / negative-path inputs.
#
# These grade the agent's ERROR HANDLING rather than output quality: given a
# malformed or unsafe submission, does the agent fail *correctly* — right
# outcome, right message, no data leak — instead of crashing, hanging, or
# silently fabricating?
#
# All cases below short-circuit BEFORE any LLM call (input validation at
# /api/submit, or the NODE 0 PII guardrail), so they run with no judge model
# and incur no OpenAI billing — safe for the always-on CI gate.
# ─────────────────────────────────────────────────────────────────────────────

# A minimal valid JD reused across cases that only vary the resume.
_JD_BACKEND = GOLDEN_EXAMPLES[0].job_description
_RESUME_CLEAN = (
    "Priya Sharma\n"
    "Backend engineer with 5 years building Python services.\n"
    "Skills: Python, FastAPI, PostgreSQL, Kafka\n"
)

ADVERSARIAL_EXAMPLES: list[Example] = [
    # Sensitive PII (SSN) must halt the run before anything reaches the LLM,
    # report a clear reason, and never echo the raw value back.
    Example(
        name="ssn_in_resume",
        job_description=_JD_BACKEND,
        resume_text=_RESUME_CLEAN + "SSN: 123-45-6789\n",
        expect="error",
        expect_message_contains="Sensitive PII",
        must_not_leak="123-45-6789",
    ),
    # Empty / whitespace-only inputs must be rejected at the door (HTTP 400),
    # not started as a doomed background run.
    Example(
        name="empty_resume",
        job_description=_JD_BACKEND,
        resume_text="   ",
        expect="reject_400",
        expect_message_contains="Resume text is required",
    ),
    Example(
        name="empty_jd",
        job_description="",
        resume_text=_RESUME_CLEAN,
        expect="reject_400",
        expect_message_contains="Job description is required",
    ),
]

