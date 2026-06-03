# Autonomous Job Application Agent
### Agentic AI Capstone Project — LangGraph + GPT-4o + FastAPI

## What it does

Given a **job description** and your **resume**, this multi-agent system autonomously:

1. **Guards PII** — scans and redacts sensitive data before anything reaches an LLM; blocks on high-risk PII (SSN, credit card, passport)
2. **Researches** the target company using web search
3. **Analyzes** the JD for ATS keywords, required skills, and tone
4. **Tailors your resume** to match the role (no fabrication — only reframing)
5. **Writes a cover letter** that references real company details
6. **Generates interview prep** — technical Qs, STAR behavioural Qs, questions to ask
7. **Quality-checks** all outputs with an LLM critic agent
8. **Loops for human review** — approve or give per-document feedback to regenerate

A **FastAPI web UI** drives the whole flow: paste or upload (PDF) the job
description and resume, watch each agent node complete live, give feedback per
card, and export any generated document to PDF.

## Setup

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install & run

```bash
cd autonomous_job_application_agent

# Install all dependencies (creates .venv automatically)
uv sync

# Add your OpenAI key to the .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

### Start the web server

The FastAPI app is the primary entry point. Run uvicorn from the **parent
directory** so the `autonomous_job_application_agent` package resolves:

```bash
cd /Users/prabrisha/agentic_ai/aiengg/capstone_project

uv run --project autonomous_job_application_agent \
  uvicorn autonomous_job_application_agent.api:app --reload --port 8000
```

Then open <http://localhost:8000> in your browser.

`--reload` restarts the server on code changes (drop it in production).

### Run the CLI (optional)

A headless command-line runner with the same human-in-the-loop flow:

```bash
cd autonomous_job_application_agent
uv run python -m autonomous_job_application_agent.main
```

### Dependency management

| Task | Command |
|---|---|
| Add a package | `uv add <package>` |
| Remove a package | `uv remove <package>` |
| Update all deps | `uv lock --upgrade` |
| Sync env after pulling | `uv sync` |

## Project structure

```
autonomous_job_application_agent/
├── __init__.py
├── api.py               ← FastAPI server (web UI, SSE-style polling, PDF I/O)
├── main.py              ← CLI runner + HITL loop
├── graph.py             ← LangGraph StateGraph
├── state.py             ← Shared AgentState TypedDict
├── pyproject.toml       ← Project metadata & dependencies (uv)
├── uv.lock              ← Reproducible lockfile (commit this)
├── .python-version      ← Pinned Python version for uv
├── requirements.txt     ← Legacy reference (pyproject.toml is authoritative)
├── README.md
├── agents/
│   ├── __init__.py
│   └── nodes.py         ← All agent node functions
├── tools/
│   ├── __init__.py
│   └── search_tools.py  ← @tool-decorated tools
├── guardrails/
│   ├── __init__.py
│   └── pii_guard.py     ← PII scan / redact / restore + block on sensitive PII
├── utils/
│   ├── __init__.py
│   ├── pdf_parser.py    ← Extract text from uploaded PDFs (pypdf)
│   └── pdf_export.py    ← Render markdown → PDF for download (fpdf2)
├── static/
│   └── index.html       ← Single-page web UI
└── outputs/             ← Generated docs saved here on approval
```

## LangGraph concepts covered

| Concept | Where |
|---|---|
| `StateGraph` + `TypedDict` | `state.py`, `graph.py` |
| Sequential edges | `parse_input → research → analyze_jd` |
| Parallel fan-out/fan-in | `analyze_jd → [3 nodes] → aggregate` |
| Conditional edges | `route_after_aggregate`, `route_after_human` |
| HITL interrupt | `interrupt_before=["human_review"]` |
| `MemorySaver` checkpointer | Persists state across interrupt/resume |
| `@tool` decorator | `tools/search_tools.py` |
| PII guardrail before LLM calls | `guardrails/pii_guard.py` |

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serve the single-page web UI |
| `POST` | `/api/submit` | Start a run; returns a `session_id` |
| `GET` | `/api/status/{id}` | Poll node-completion events (cursor-based) |
| `POST` | `/api/feedback/{id}` | Approve or send per-document feedback |
| `POST` | `/api/parse-pdf` | Upload a PDF, get extracted text back |
| `POST` | `/api/export-pdf` | Render markdown content to a downloadable PDF |

## Extension ideas

- Swap DuckDuckGo for **Tavily API** for richer search results
- Enable **LangSmith tracing** for observability
- Use **SQLite checkpointer** for state that survives restarts
- Replace the in-memory session store with **Redis** for multi-worker deploys
- Multi-company mode: process 5 JDs in parallel with `asyncio`
