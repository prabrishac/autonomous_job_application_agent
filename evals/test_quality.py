"""
test_quality.py — pytest gate for CI, built on the deepeval metrics.

Reads the cached outputs in evals/fixtures/ (produced by `evaluate.py`) and
asserts each metric passes its threshold. Fixtures are used rather than live
API calls so CI is deterministic and cheap; generate/refresh them first with:

    cd <project parent>
    python -m autonomous_job_application_agent.evals.evaluate --refresh

Then gate in CI with:

    pytest autonomous_job_application_agent/evals/test_quality.py

deepeval's judge model still calls OpenAI, so OPENAI_API_KEY must be set.
"""

import json
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deepeval import assert_test

_PKG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PKG_ROOT / ".env")

from autonomous_job_application_agent.evals.dataset import GOLDEN_EXAMPLES
from autonomous_job_application_agent.evals import metrics as M

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    cache = _FIXTURES / f"{name}.json"
    if not cache.exists():
        pytest.skip(f"No cached outputs for '{name}'. Run evaluate.py --refresh first.")
    return json.loads(cache.read_text())


@pytest.mark.parametrize("ex", GOLDEN_EXAMPLES, ids=lambda e: e.name)
def test_no_stray_pii_tokens(ex):
    out = _load(ex.name)
    leaks = M.check_no_pii_tokens(
        out.get("tailored_resume", ""),
        out.get("cover_letter", ""),
        out.get("interview_questions", ""),
    )
    assert not leaks, f"Stray PII tokens leaked into final docs: {sorted(set(leaks))}"


@pytest.mark.parametrize("ex", GOLDEN_EXAMPLES, ids=lambda e: e.name)
def test_resume_metrics(ex):
    out = _load(ex.name)
    tc = M.resume_test_case(ex.job_description, ex.resume_text, out.get("tailored_resume", ""))
    assert_test(tc, [
        M.no_fabrication_metric(),
        M.faithfulness_metric(),
        M.jd_alignment_metric(),
    ])


@pytest.mark.parametrize("ex", GOLDEN_EXAMPLES, ids=lambda e: e.name)
def test_cover_letter_quality(ex):
    out = _load(ex.name)
    tc = M.cover_letter_test_case(ex.job_description, out.get("cover_letter", ""))
    assert_test(tc, [M.cover_letter_quality_metric()])
