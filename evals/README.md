# Evaluations (deepeval)

LLM-as-judge evaluation for the agent's generated documents, run through the
**real API** (`api.py`) — `POST /api/submit` → poll `/api/status` → grab the
outputs at the human-review interrupt. This exercises the exact handlers and
background-thread graph that `uvicorn ...api:app` serves.

## What it measures

| Metric | Type | Checks |
|---|---|---|
| **No Fabrication** | GEval | Tailored resume invents no employer/title/date/metric/skill beyond the original resume (the system's core guarantee). |
| **Faithfulness** | deepeval built-in | Tailored resume claims are grounded in the original resume (used as context). |
| **JD Alignment** | GEval | Tailored resume surfaces the JD's required skills/ATS keywords the candidate genuinely has. |
| **Cover Letter Quality** | GEval | Specific hook, real experience tied to the role, no banned clichés, concise. |
| **No stray PII tokens** | deterministic | Final docs contain no leftover `[PII:*]` redaction tokens. |

Golden inputs live in [dataset.py](dataset.py). Resumes are intentionally
narrower than the JD so fabrication has something to catch.

## Error-handling evals (negative paths)

The metrics above grade output *quality* on clean inputs. A separate suite grades
the agent's *behavior on bad or unsafe inputs* — does it fail **correctly**?

[test_error_handling.py](test_error_handling.py) submits adversarial cases from
`ADVERSARIAL_EXAMPLES` in [dataset.py](dataset.py) and asserts the right failure
mode, a useful message, and **no data leak**:

| Case | Expected outcome | Asserts |
|---|---|---|
| **ssn_in_resume** | `status=error` | Halts at the NODE 0 PII guard before any LLM call; message says "Sensitive PII"; the raw SSN is never echoed back; no output payload produced. |
| **empty_resume** | `HTTP 400` | Rejected at `/api/submit` with "Resume text is required". |
| **empty_jd** | `HTTP 400` | Rejected at `/api/submit` with "Job description is required". |

Each case short-circuits **before any LLM call**, so this whole file runs
deterministically with **no judge model and no OpenAI billing** — safe on every
push, no fixture refresh needed:

```bash
pytest evals/test_error_handling.py        # ~4s, free
```

`run_via_api()` in [api_client.py](api_client.py) is the harness used here: unlike
`generate_via_api()` (which raises), it classifies the outcome
(`success` / `error` / `reject_400` / `timeout`) and returns it for assertion.

**Known-gap markers (`xfail`).** Two tests assert *desired* PII behavior the guard
doesn't yet have — an undashed SSN (`123456789`) and an SSN in the **job
description** (the guard only scans the resume). They're marked `xfail`; when the
guard is hardened, they flip to an unexpected pass — the signal to promote them to
hard asserts.

The live quality runner ([evaluate.py](evaluate.py)) is also error-resilient: if
one example fails to generate, it's recorded as a run failure and the suite keeps
scoring the rest (and still exits non-zero) instead of aborting on the first error.

## Latency (per-node)

The suite also records **per-node latency** so you can see where a run spends its
time (web search vs. each LLM node) and gate on it.

How it works:
- `api.py`'s `_push` stamps every event with a monotonic server clock (`t`).
- `node_latencies()` in [api_client.py](api_client.py) diffs consecutive events to
  attribute wall-clock time to each node (`pii_guardrail`, `research`,
  `tailor_resume`, …), measured up to the first human-review interrupt.
- The breakdown is stored in the fixture as `_node_latency` (+ `_latency_s` total)
  at generation time, so it survives caching.

`evaluate.py` prints the breakdown and flags any node over budget:

```
· latency: 78.4s total (budget 60s/node)
    ✗ research               64.20s
      tailor_resume          11.30s
      ...
```

`check_node_latency()` in [metrics.py](metrics.py) is the deterministic gate (no
judge). Budget defaults to **60s/node**; override with `EVAL_MAX_NODE_LATENCY_S`.
[test_quality.py](test_quality.py)'s `test_node_latency` gates the stored value in
CI (and **skips** fixtures generated before latency was captured).

⚠️ Caveats — be honest about what the number means:
- **Latency "as of last fixture refresh."** Cached runs do no API work; the gate
  reads the value stored at the last `--refresh`. Re-`--refresh` to update it.
- **In-process ≠ production.** The default `TestClient` path has no uvicorn/network
  overhead; use `JOB_AGENT_BASE_URL` for production-representative timing.
- **Single sample is noisy.** One run isn't a percentile — keep the budget generous
  (the 60s default) so CI doesn't flake. For rigor, refresh N times and gate on p95.

## Cost (per-node tokens)

The suite also records **token usage** per run and per node, so you can see what a
run costs and gate on it.

How it works:
- `UsageTracker` (a `BaseCallbackHandler` in [api.py](../api.py)) is attached via
  `config["callbacks"]`. LangGraph propagates it to every LLM call and tags each
  with its `langgraph_node`, so the tracker accumulates **total + per-node** tokens
  with no changes to the node code.
- Totals are merged into the output payload and stored in the fixture as `_tokens`
  (`{input, output, total}`) and `_node_tokens` (per node), so they survive caching.

`evaluate.py` prints the breakdown and an estimated cost:

```
· cost: 24130 tokens (~$0.0098; budget 80000 tok, $ unpriced-gate-off)
    write_cover_letter      7200 tok  (in 5400 / out 1800)
    tailor_resume           6800 tok  (in 5100 / out 1700)
    ...
```

Gating is **token-first** because token counts are exact, while dollars depend on a
price the configured model (`gpt-5.4-mini`) doesn't expose:

- **Tokens** — always gated. Default budget **80 000 tokens/run**; override with
  `EVAL_MAX_TOKENS`.
- **Dollars** — only gated when `EVAL_MAX_COST_USD` is set. The $ estimate uses
  `EVAL_PRICE_PER_1M_INPUT` / `EVAL_PRICE_PER_1M_OUTPUT` (defaults 0.15 / 0.60);
  set these to your model's real rate or the dollar figure is meaningless.

[test_quality.py](test_quality.py)'s `test_cost` gates the stored usage in CI and
**skips** fixtures generated before usage was captured.

⚠️ Same caveats as latency: it's usage **"as of last refresh"** (cached runs do no
API work), a single sample varies run-to-run, and the price table is a config value
you own — keep budgets generous so CI doesn't flake. **Agent vs judge cost:** this
measures only the agent's own LLM calls, not the deepeval judge calls (which are a
cost of *running* the evals, billed separately when the judge model runs).

## Setup

```bash
pip install -r requirements-dev.txt      # deepeval + pytest
# OPENAI_API_KEY must be in .env (used by both the agent and the judge model)
```

## Run

Run from the project's **parent** directory (same convention as `main`/`api`):

```bash
cd /Users/prabrisha/agentic_ai/aiengg/capstone_project

# Generate outputs via the API + score them (caches to evals/fixtures/)
python -m autonomous_job_application_agent.evals.evaluate --refresh

# Re-score cached outputs without paying for another agent run
python -m autonomous_job_application_agent.evals.evaluate

# Score against a live uvicorn server instead of the in-process app
JOB_AGENT_BASE_URL=http://localhost:8000 \
  python -m autonomous_job_application_agent.evals.evaluate --refresh
```

`evaluate.py` prints a scorecard and exits non-zero if any metric is below
threshold or a PII token leaks.

## Fabrication regression controls

[test_fabrication_regression.py](test_fabrication_regression.py) pins the
No-Fabrication detectors themselves — no live agent run, just the judge. For
**each of the three outputs** (tailored resume, cover letter, interview prep
guide) there are three controls against the same base resume:

- **positive control** — a blatantly fabricated document (invented employer,
  degree, certification, leadership, inflated tenure) must score *below* threshold;
- **negative control** — an honest version using only the real resume must *pass*;
- the fabricated document must score strictly *below* the faithful one.

The interview-prep case also confirms that generic technical Q&A is *not*
penalised as fabrication — only false first-person claims about the candidate's
own history are.

```bash
pytest autonomous_job_application_agent/evals/test_fabrication_regression.py
```

These guard against a future prompt/model change making a detector blind to
fabrication or trigger-happy on legitimate reframing. Dedicated metrics back each
output: `no_fabrication_metric` (resume), `cover_letter_no_fabrication_metric`,
and `interview_prep_no_fabrication_metric` in [metrics.py](metrics.py).

## CI gate

```bash
# 1. refresh fixtures once (live agent run)
python -m autonomous_job_application_agent.evals.evaluate --refresh
# 2. gate on cached fixtures (judge still calls OpenAI)
pytest autonomous_job_application_agent/evals/test_quality.py
```

Tuning thresholds: edit the `threshold=` values in [metrics.py](metrics.py).
Override the judge model with `EVAL_MODEL=gpt-4o-mini` for cheaper runs.
