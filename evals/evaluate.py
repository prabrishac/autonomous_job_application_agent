"""
evaluate.py — Run the deepeval suite over the golden dataset.

Pipeline per example:
  1. Generate outputs through the real API (api_client.generate_via_api),
     caching them under evals/fixtures/<name>.json so re-runs are free.
  2. Score the tailored resume and cover letter with the deepeval metrics.
  3. Run the deterministic PII-token leak check on the final documents.
  4. Print a scorecard and exit non-zero if any metric is below threshold.

Run from the project ROOT directory (same convention as main/api):

    cd /Users/prabrisha/agentic_ai/aiengg/capstone_project/autonomous_job_application_agent
    python -m evals.evaluate            # use cache
    python -m evals.evaluate --refresh  # regenerate
    JOB_AGENT_BASE_URL=http://localhost:8000 python -m evals.evaluate # live server

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

from evals.dataset import GOLDEN_EXAMPLES, Example
from evals.api_client import generate_via_api
from evals import metrics as M

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


def _report_latency(outputs: dict) -> list[tuple[str, float]]:
    """Print the per-node latency breakdown and return any nodes over budget."""
    per_node = outputs.get("_node_latency") or {}
    if not per_node:
        print("  · latency: not captured (refresh fixtures to record it)")
        return []
    total = outputs.get("_latency_s", sum(per_node.values()))
    budget = M.node_latency_budget()
    print(f"  · latency: {total:.1f}s total (budget {budget:.0f}s/node)")
    for node, secs in sorted(per_node.items(), key=lambda x: x[1], reverse=True):
        mark = "✗" if secs > budget else " "
        print(f"      {mark} {node:<22} {secs:6.2f}s")
    return M.check_node_latency(per_node)


def _report_cost(outputs: dict) -> bool:
    """Print token usage + per-node breakdown and est. cost. Return True if over budget."""
    tokens = outputs.get("_tokens") or {}
    per_node = outputs.get("_node_tokens") or {}
    if not tokens:
        print("  · cost: not captured (refresh fixtures to record it)")
        return False
    total = tokens.get("total") or tokens.get("input", 0) + tokens.get("output", 0)
    cost = M.estimate_cost_usd(tokens)
    cap = M.cost_budget_usd()
    cap_str = f", ${cap:.4f} cap" if cap is not None else ", $ unpriced-gate-off"
    print(f"  · cost: {total} tokens (~${cost:.4f}; budget {M.token_budget()} tok{cap_str})")
    for node, t in sorted(per_node.items(), key=lambda x: (x[1].get("total") or x[1]["input"] + x[1]["output"]), reverse=True):
        nt = t.get("total") or t["input"] + t["output"]
        print(f"      {node:<22} {nt:6d} tok  (in {t.get('input',0)} / out {t.get('output',0)})")
    over_tokens = M.check_token_budget(tokens)
    over_cost = M.check_cost_budget(tokens)
    if over_tokens:
        print(f"  ✗ over token budget: {over_tokens} > {M.token_budget()}")
    if over_cost:
        print(f"  ✗ over cost budget: ${over_cost:.4f} > ${cap:.4f}")
    return bool(over_tokens or over_cost)


def _score_example(ex: Example, outputs: dict) -> tuple[list[dict], bool, list[tuple[str, float]], bool]:
    """Score one example's outputs. Returns (rows, pii_leaked, slow_nodes, over_cost)."""
    tailored_resume = outputs.get("tailored_resume", "")
    cover_letter = outputs.get("cover_letter", "")
    interview = outputs.get("interview_questions", "")
    print(f"  · quality_score (agent self-report): {outputs.get('quality_score', 0.0)}")

    slow_nodes = _report_latency(outputs)
    over_cost = _report_cost(outputs)

    # Deterministic PII-leak guard on the final documents.
    leaks = M.check_no_pii_tokens(tailored_resume, cover_letter, interview)
    if leaks:
        print(f"  ✗ PII token leak: {sorted(set(leaks))}")
    else:
        print("  ✓ no stray PII tokens")

    resume_tc = M.resume_test_case(ex.job_description, ex.resume_text, tailored_resume)
    cover_tc = M.cover_letter_test_case(ex.job_description, cover_letter)

    rows: list[dict] = []
    for metric, tc in [
        (M.no_fabrication_metric(), resume_tc),
        (M.faithfulness_metric(), resume_tc),
        (M.jd_alignment_metric(), resume_tc),
        (M.cover_letter_quality_metric(), cover_tc),
    ]:
        row = _score(metric, tc)
        rows.append(row)
        mark = "✓" if row["passed"] else "✗"
        print(f"  {mark} {row['name']:<22} {row['score']:.2f} (≥{row['threshold']:.2f})")
        if not row["passed"] and row["reason"]:
            print(f"      ↳ {row['reason'][:300]}")

    return rows, bool(leaks), slow_nodes, over_cost


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
    run_failures: list[str] = []
    latency_failures: list[str] = []
    cost_failures: list[str] = []

    for ex in examples:
        print(f"\n▶ {ex.name}")
        try:
            outputs = _get_outputs(ex, refresh=args.refresh)
        except Exception as exc:
            # A failed generation for one example must not abort the whole suite —
            # record it as a failure and keep scoring the rest.
            run_failures.append(ex.name)
            print(f"  ✗ run errored: {exc}")
            continue

        rows, pii_leaked, slow_nodes, over_cost = _score_example(ex, outputs)
        all_rows.extend((ex.name, r) for r in rows)
        if pii_leaked:
            pii_failures.append(ex.name)
        if slow_nodes:
            latency_failures.append(ex.name)
            slow = ", ".join(f"{n} {s:.0f}s" for n, s in slow_nodes)
            print(f"  ✗ node(s) over latency budget: {slow}")
        if over_cost:
            cost_failures.append(ex.name)

    # ── Summary ─────────────────────────────────────────────────────────────────
    metric_failures = [(name, r) for name, r in all_rows if not r["passed"]]
    print("\n" + "═" * 60)
    print(f"Metrics: {len(all_rows) - len(metric_failures)}/{len(all_rows)} passed"
          f" across {len(examples)} example(s).")
    if run_failures:
        print(f"Run errors (no outputs scored): {', '.join(run_failures)}")
    if pii_failures:
        print(f"PII leaks in: {', '.join(pii_failures)}")
    if latency_failures:
        print(f"Latency over budget in: {', '.join(latency_failures)}")
    if cost_failures:
        print(f"Cost over budget in: {', '.join(cost_failures)}")
    if metric_failures:
        print("Failures:")
        for name, r in metric_failures:
            print(f"  - [{name}] {r['name']} {r['score']:.2f} < {r['threshold']:.2f}")

    return 1 if (metric_failures or pii_failures or run_failures
                 or latency_failures or cost_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
