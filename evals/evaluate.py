"""
evaluate.py — Run the deepeval suite over the golden dataset.

Pipeline per example:
  1. Generate outputs through the real API (api_client.generate_via_api),
     caching them under evals/fixtures/<name>.json so re-runs are free.
  2. Score the tailored resume and cover letter with the deepeval metrics.
  3. Run the deterministic PII-token leak check on the final documents.
  4. Print a scorecard and exit non-zero if any metric is below threshold.

Run from the project's PARENT directory (same convention as main/api):

    cd /Users/prabrisha/agentic_ai/aiengg/capstone_project
    python -m autonomous_job_application_agent.evals.evaluate            # use cache
    python -m autonomous_job_application_agent.evals.evaluate --refresh  # regenerate
    JOB_AGENT_BASE_URL=http://localhost:8000 python -m ...evals.evaluate # live server

Requires OPENAI_API_KEY (read from .env). deepeval's judge model also calls
OpenAI, so a run bills both the agent and the judge.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_PKG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PKG_ROOT / ".env")

from autonomous_job_application_agent.evals.dataset import GOLDEN_EXAMPLES, Example
from autonomous_job_application_agent.evals.api_client import generate_via_api
from autonomous_job_application_agent.evals import metrics as M

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _get_outputs(ex: Example, *, refresh: bool) -> dict:
    """Return generated outputs for an example, using the fixture cache when possible."""
    cache = _FIXTURES / f"{ex.name}.json"
    if cache.exists() and not refresh:
        print(f"  · using cached outputs ({cache.name})")
        return json.loads(cache.read_text())

    print("  · generating via API …")
    outputs = generate_via_api(ex.job_description, ex.resume_text)
    _FIXTURES.mkdir(exist_ok=True)
    cache.write_text(json.dumps(outputs, indent=2))
    return outputs


def _score(metric, test_case) -> dict:
    """Measure one metric, returning a flat result row (robust across deepeval versions)."""
    metric.measure(test_case)
    return {
        "name": metric.__class__.__name__ if not hasattr(metric, "name") else metric.name,
        "score": round(float(metric.score), 3),
        "threshold": float(metric.threshold),
        "passed": bool(metric.is_successful()),
        "reason": (getattr(metric, "reason", "") or "").strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the job-application agent eval suite.")
    parser.add_argument("--refresh", action="store_true",
                        help="Regenerate outputs through the API instead of using cached fixtures.")
    parser.add_argument("--only", default="",
                        help="Run only the example with this name slug.")
    args = parser.parse_args()

    examples = [e for e in GOLDEN_EXAMPLES if not args.only or e.name == args.only]
    if not examples:
        print(f"No example named '{args.only}'.", file=sys.stderr)
        return 2

    all_rows: list[tuple[str, dict]] = []
    pii_failures: list[str] = []

    for ex in examples:
        print(f"\n▶ {ex.name}")
        outputs = _get_outputs(ex, refresh=args.refresh)

        tailored_resume = outputs.get("tailored_resume", "")
        cover_letter = outputs.get("cover_letter", "")
        interview = outputs.get("interview_questions", "")
        print(f"  · quality_score (agent self-report): {outputs.get('quality_score', 0.0)}")

        # Deterministic PII-leak guard on the final documents.
        leaks = M.check_no_pii_tokens(tailored_resume, cover_letter, interview)
        if leaks:
            pii_failures.append(ex.name)
            print(f"  ✗ PII token leak: {sorted(set(leaks))}")
        else:
            print("  ✓ no stray PII tokens")

        resume_tc = M.resume_test_case(ex.job_description, ex.resume_text, tailored_resume)
        cover_tc = M.cover_letter_test_case(ex.job_description, cover_letter)

        for metric, tc in [
            (M.no_fabrication_metric(), resume_tc),
            (M.faithfulness_metric(), resume_tc),
            (M.jd_alignment_metric(), resume_tc),
            (M.cover_letter_quality_metric(), cover_tc),
        ]:
            row = _score(metric, tc)
            all_rows.append((ex.name, row))
            mark = "✓" if row["passed"] else "✗"
            print(f"  {mark} {row['name']:<22} {row['score']:.2f} (≥{row['threshold']:.2f})")
            if not row["passed"] and row["reason"]:
                print(f"      ↳ {row['reason'][:300]}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    metric_failures = [(name, r) for name, r in all_rows if not r["passed"]]
    print("\n" + "═" * 60)
    print(f"Metrics: {len(all_rows) - len(metric_failures)}/{len(all_rows)} passed"
          f" across {len(examples)} example(s).")
    if pii_failures:
        print(f"PII leaks in: {', '.join(pii_failures)}")
    if metric_failures:
        print("Failures:")
        for name, r in metric_failures:
            print(f"  - [{name}] {r['name']} {r['score']:.2f} < {r['threshold']:.2f}")

    return 1 if (metric_failures or pii_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
