"""
test_quality.py — pytest gate for CI, built on the deepeval metrics.

Reads the cached outputs in evals/fixtures/ (produced by `evaluate.py`) and
asserts each metric passes its threshold. Fixtures are used rather than live
API calls so CI is deterministic and cheap; generate/refresh them first with:

    cd <project root>
    python -m evals.evaluate --refresh

Then gate in CI with:

    pytest evals/test_quality.py

deepeval's judge model still calls OpenAI, so OPENAI_API_KEY must be set.
"""

import json
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deepeval import assert_test

_PKG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PKG_ROOT / ".env")

from evals.dataset import GOLDEN_EXAMPLES
from evals import metrics as M

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


@pytest.mark.parametrize("ex", GOLDEN_EXAMPLES, ids=lambda e: e.name)
def test_node_latency(ex):
    """
    Gate per-node latency recorded at the last fixture refresh. This reads the
    stored `_node_latency` (latency "as of last refresh"), not a live timing, so
    it stays deterministic in CI. Skips for fixtures generated before latency was
    captured. Tune the budget with EVAL_MAX_NODE_LATENCY_S.
    """
    out = _load(ex.name)
    per_node = out.get("_node_latency")
    if not per_node:
        pytest.skip("No latency recorded for this fixture. Run evaluate.py --refresh.")
    slow = M.check_node_latency(per_node)
    assert not slow, (
        f"Node(s) over {M.node_latency_budget():.0f}s budget: "
        + ", ".join(f"{n}={s:.1f}s" for n, s in slow)
    )


@pytest.mark.parametrize("ex", GOLDEN_EXAMPLES, ids=lambda e: e.name)
def test_cost(ex):
    """
    Gate the agent's token usage recorded at the last fixture refresh. Primary
    gate is on TOKENS (exact); the $ gate is only enforced when EVAL_MAX_COST_USD
    is set, since the configured model's price isn't known here. Skips fixtures
    generated before usage was captured. Tune with EVAL_MAX_TOKENS.
    """
    out = _load(ex.name)
    tokens = out.get("_tokens")
    if not tokens:
        pytest.skip("No token usage recorded for this fixture. Run evaluate.py --refresh.")
    over_tokens = M.check_token_budget(tokens)
    assert not over_tokens, f"Run used {over_tokens} tokens > {M.token_budget()} budget."
    over_cost = M.check_cost_budget(tokens)
    assert not over_cost, f"Run cost ~${over_cost:.4f} > ${M.cost_budget_usd():.4f} budget."
