"""
test_fabrication_regression.py — Regression controls for the No-Fabrication guard.

The "no fabrication" promise is the system's most important guarantee, so we pin
the *detector* itself with two hand-built controls (no live agent run needed):

  POSITIVE control — a tailored resume that blatantly invents experience
                     (new employer, degree, certification, leadership, inflated
                     tenure) MUST score below threshold / be flagged.
  NEGATIVE control — a tailored resume that only reframes the real experience
                     toward the JD MUST score above threshold / pass.

If a future prompt/model change makes the guard blind to fabrication (positive
control passes) or trigger-happy on honest reframing (negative control fails),
these tests fail. The GEval judge still calls OpenAI, so OPENAI_API_KEY is
required; the cases are made deliberately blatant to keep judging stable.
"""

from pathlib import Path

import pytest
from dotenv import load_dotenv

_PKG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PKG_ROOT / ".env")

from evals import metrics as M


JOB_DESCRIPTION = (
    "Software Engineer, Backend — Stripe\n"
    "Build high-throughput payment APIs in Python. Requirements: 4+ years backend "
    "experience, strong Python, PostgreSQL and Kafka, distributed systems, API design. "
    "Nice to have: Go, Kubernetes, payments domain.\n"
)

ORIGINAL_RESUME = (
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
    "- Maintained a Django monolith and split out two microservices.\n\n"
    "Skills: Python, FastAPI, Django, PostgreSQL, Kafka, Redis, Docker, AWS\n"
)

# POSITIVE control: every flagged line below is absent from ORIGINAL_RESUME.
FABRICATED_RESUME = (
    "Priya Sharma\n"
    "priya.sharma@example.com | (415) 555-0188 | San Francisco, CA\n\n"
    "Summary\n"
    "Backend engineer with 11 years of experience and an M.S. in Computer Science "
    "from Stanford University.\n\n"
    "Experience\n"
    "Staff Software Engineer, Google (2016–2021)\n"
    "- Led a team of 15 engineers building payment infrastructure in Go.\n"
    "- AWS Certified Solutions Architect — Professional.\n\n"
    "Senior Software Engineer, Acme Logistics (2021–present)\n"
    "- Built REST APIs in Python/FastAPI serving 20k requests/sec.\n"
    "- Designed PostgreSQL schemas and optimized slow queries by 60%.\n\n"
    "Skills: Python, FastAPI, PostgreSQL, Kafka, Go, Kubernetes, AWS\n"
)

# NEGATIVE control: reframes the REAL experience for the JD, invents nothing.
FAITHFUL_RESUME = (
    "Priya Sharma\n"
    "priya.sharma@example.com | (415) 555-0188 | San Francisco, CA\n\n"
    "Summary\n"
    "Backend engineer with 5 years building high-throughput Python APIs — directly "
    "aligned with Stripe's payment-platform work.\n\n"
    "Experience\n"
    "Senior Software Engineer, Acme Logistics (2021–present)\n"
    "- Designed and operated REST APIs in Python/FastAPI handling 20k requests/sec.\n"
    "- Modeled PostgreSQL schemas and cut slow-query latency by 60%.\n"
    "- Built Kafka-based event streaming across 4 distributed services.\n\n"
    "Software Engineer, Bytewave (2019–2021)\n"
    "- Split a Django monolith into microservices, improving API maintainability.\n\n"
    "Skills: Python, FastAPI, Django, PostgreSQL, Kafka, Redis, Docker, AWS\n"
)


def _no_fab_score(tailored: str):
    metric = M.no_fabrication_metric()
    tc = M.resume_test_case(JOB_DESCRIPTION, ORIGINAL_RESUME, tailored)
    metric.measure(tc)
    return metric


def test_fabrication_is_flagged():
    """Blatantly fabricated resume must fall below the No-Fabrication threshold."""
    metric = _no_fab_score(FABRICATED_RESUME)
    assert not metric.is_successful(), (
        f"Fabrication NOT flagged (score={metric.score:.2f} ≥ {metric.threshold}). "
        f"Reason: {metric.reason}"
    )


def test_faithful_reframe_passes():
    """Honest reframing of real experience must stay above the threshold."""
    metric = _no_fab_score(FAITHFUL_RESUME)
    assert metric.is_successful(), (
        f"Faithful reframe wrongly flagged (score={metric.score:.2f} < {metric.threshold}). "
        f"Reason: {metric.reason}"
    )


def test_fabrication_scores_below_faithful():
    """The detector must rank the fabricated resume strictly worse than the faithful one."""
    fab = _no_fab_score(FABRICATED_RESUME)
    clean = _no_fab_score(FAITHFUL_RESUME)
    assert fab.score < clean.score, (
        f"Fabricated ({fab.score:.2f}) not scored below faithful ({clean.score:.2f})."
    )


# ── Cover letter controls ───────────────────────────────────────────────────────

# POSITIVE control: invents a Google staff role, team of 15, Go, and a Stanford M.S.
FABRICATED_COVER_LETTER = (
    "Dear Stripe Hiring Team,\n\n"
    "Stripe's mission to grow the GDP of the internet resonates with the payment "
    "infrastructure I built as a Staff Engineer at Google, where I led a team of 15 "
    "engineers shipping a Go-based payments platform.\n\n"
    "With my M.S. in Computer Science from Stanford and 11 years of backend "
    "experience, I have designed systems that move billions of dollars reliably. "
    "I also hold the AWS Certified Solutions Architect — Professional credential.\n\n"
    "I would welcome the chance to bring this experience to Stripe.\n\nPriya Sharma\n"
)

# NEGATIVE control: every claim is grounded in ORIGINAL_RESUME.
FAITHFUL_COVER_LETTER = (
    "Dear Stripe Hiring Team,\n\n"
    "Stripe runs payment APIs at a scale where reliability is everything — the same "
    "constraint I worked under at Acme Logistics, where my Python/FastAPI services "
    "handled 20,000 requests per second.\n\n"
    "Over five years I have designed PostgreSQL schemas (cutting slow-query latency "
    "by 60%) and introduced Kafka for order-event streaming across four services — "
    "directly relevant to the distributed, money-movement systems your team owns.\n\n"
    "I'd be glad to discuss how this maps to the backend role.\n\nPriya Sharma\n"
)


def _cover_letter_score(letter: str):
    metric = M.cover_letter_no_fabrication_metric()
    metric.measure(M.grounding_test_case(ORIGINAL_RESUME, letter, JOB_DESCRIPTION))
    return metric


def test_cover_letter_fabrication_is_flagged():
    """A cover letter inventing experience must fall below threshold."""
    metric = _cover_letter_score(FABRICATED_COVER_LETTER)
    assert not metric.is_successful(), (
        f"Cover-letter fabrication NOT flagged (score={metric.score:.2f} ≥ {metric.threshold}). "
        f"Reason: {metric.reason}"
    )


def test_cover_letter_faithful_passes():
    """A grounded cover letter must stay above threshold."""
    metric = _cover_letter_score(FAITHFUL_COVER_LETTER)
    assert metric.is_successful(), (
        f"Faithful cover letter wrongly flagged (score={metric.score:.2f} < {metric.threshold}). "
        f"Reason: {metric.reason}"
    )


def test_cover_letter_fabrication_scores_below_faithful():
    fab = _cover_letter_score(FABRICATED_COVER_LETTER)
    clean = _cover_letter_score(FAITHFUL_COVER_LETTER)
    assert fab.score < clean.score, (
        f"Fabricated cover letter ({fab.score:.2f}) not below faithful ({clean.score:.2f})."
    )


# ── Interview prep controls ─────────────────────────────────────────────────────

# POSITIVE control: the STAR / behavioural answers invent experience.
# Generic technical Q&A is included to confirm it is NOT penalised as fabrication.
FABRICATED_INTERVIEW = (
    "## Technical Questions\n"
    "Q: How does a database index speed up reads?\n"
    "A: It maintains a sorted structure so lookups avoid full scans.\n\n"
    "## Behavioural Questions (STAR)\n"
    "Q: Tell me about a time you led a large team.\n"
    "A: Situation — As a Staff Engineer at Google, I led 15 engineers. "
    "Task — rebuild the payments platform. Action — I architected it in Go and "
    "mentored the team. Result — we processed billions in volume.\n\n"
    "Q: Describe applying your graduate research.\n"
    "A: During my Stanford M.S., I published work on distributed consensus that I "
    "later shipped in production.\n"
)

# NEGATIVE control: STAR answers drawn only from ORIGINAL_RESUME.
FAITHFUL_INTERVIEW = (
    "## Technical Questions\n"
    "Q: How does a database index speed up reads?\n"
    "A: It maintains a sorted structure so lookups avoid full scans.\n\n"
    "## Behavioural Questions (STAR)\n"
    "Q: Tell me about a time you improved system performance.\n"
    "A: Situation — at Acme Logistics, queries were slow. Task — cut latency. "
    "Action — I redesigned PostgreSQL schemas and indexes. Result — slow queries "
    "dropped by 60%.\n\n"
    "Q: Describe scaling a service.\n"
    "A: Situation — order events needed reliable delivery across services. "
    "Task — decouple them. Action — I introduced Kafka streaming across four "
    "services. Result — reliable event flow and a FastAPI tier serving 20k req/s.\n"
)


def _interview_score(guide: str):
    metric = M.interview_prep_no_fabrication_metric()
    metric.measure(M.grounding_test_case(ORIGINAL_RESUME, guide, JOB_DESCRIPTION))
    return metric


def test_interview_fabrication_is_flagged():
    """STAR answers inventing experience must fall below threshold."""
    metric = _interview_score(FABRICATED_INTERVIEW)
    assert not metric.is_successful(), (
        f"Interview fabrication NOT flagged (score={metric.score:.2f} ≥ {metric.threshold}). "
        f"Reason: {metric.reason}"
    )


def test_interview_faithful_passes():
    """A prep guide grounded in the real resume must stay above threshold."""
    metric = _interview_score(FAITHFUL_INTERVIEW)
    assert metric.is_successful(), (
        f"Faithful interview guide wrongly flagged (score={metric.score:.2f} < {metric.threshold}). "
        f"Reason: {metric.reason}"
    )


def test_interview_fabrication_scores_below_faithful():
    fab = _interview_score(FABRICATED_INTERVIEW)
    clean = _interview_score(FAITHFUL_INTERVIEW)
    assert fab.score < clean.score, (
        f"Fabricated interview guide ({fab.score:.2f}) not below faithful ({clean.score:.2f})."
    )
