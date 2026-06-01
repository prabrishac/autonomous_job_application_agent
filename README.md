# Autonomous Job Application Agent
### Agentic AI Capstone Project — LangGraph + GPT-4o

## What it does

Given a **job description** and your **resume**, this multi-agent system autonomously:

1. **Researches** the target company using web search
2. **Analyzes** the JD for ATS keywords, required skills, and tone
3. **Tailors your resume** to match the role (no fabrication — only reframing)
4. **Writes a cover letter** that references real company details
5. **Generates interview prep** — technical Qs, STAR behavioural Qs, questions to ask
6. **Quality-checks** all outputs with an LLM critic agent
7. **Loops for human review** — approve or give feedback to regenerate

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

# Run
uv run python -m autonomous_job_application_agent.main
```

The `.env` file is read automatically at startup — no `export` needed.

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
├── utils/
│   └── __init__.py
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

## Extension ideas

- Swap DuckDuckGo for **Tavily API** for richer search results
- Add a **Streamlit UI** to replace the CLI HITL loop
- Enable **LangSmith tracing** for observability
- Use **SQLite checkpointer** for state that survives restarts
- Multi-company mode: process 5 JDs in parallel with `asyncio`
