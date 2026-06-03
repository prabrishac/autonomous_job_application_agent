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
