"""
api_client.py — Drives the agent through its real HTTP API (api.py).

The agent's user-facing entry point is the FastAPI app, not the CLI, so the eval
exercises exactly that surface:

    POST /api/submit            → starts a background agent run, returns session_id
    GET  /api/status/{id}       → poll for events until "awaiting_feedback"
    POST /api/feedback/{id}     → (optional) approve to finalise the session

By default this runs the app *in-process* with Starlette's TestClient — the same
route handlers and background-thread graph that `uvicorn ...api:app` serves, but
with no server to start. Set JOB_AGENT_BASE_URL=http://localhost:8000 to instead
hit a live uvicorn server over HTTP.

`generate_via_api()` returns the output payload the UI receives at the
human-review interrupt: tailored_resume, cover_letter, interview_questions,
quality_score, etc.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager


# How long to wait for a single agent run to reach the human-review interrupt.
_POLL_TIMEOUT_S = 240
_POLL_INTERVAL_S = 1.5


@contextmanager
def _client():
    """Yield an HTTP-ish client with .get/.post(json=...) → response.json()."""
    base_url = os.environ.get("JOB_AGENT_BASE_URL")
    if base_url:
        import httpx
        with httpx.Client(base_url=base_url, timeout=30) as c:
            yield c
    else:
        # In-process: import here so a live-server run never needs the app deps.
        from starlette.testclient import TestClient
        from autonomous_job_application_agent.api import app
        with TestClient(app) as c:
            yield c


def _latest_state(events: list[dict]) -> dict | None:
    """Return the most recent event payload that carries a `state` snapshot."""
    for ev in reversed(events):
        if isinstance(ev, dict) and ev.get("state"):
            return ev["state"]
    return None


def generate_via_api(job_description: str, resume_text: str, *, approve: bool = True) -> dict:
    """
    Submit one (JD, resume) pair through the API and return the generated outputs.

    Polls /api/status until the run reaches the human-review interrupt
    ("awaiting_feedback"), captures the output payload, then optionally approves
    the session so it completes cleanly.

    Raises RuntimeError on agent error (e.g. PII blocked, missing API key) or
    if the run does not reach review within _POLL_TIMEOUT_S.
    """
    with _client() as client:
        resp = client.post(
            "/api/submit",
            json={"job_description": job_description, "resume_text": resume_text},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"/api/submit failed [{resp.status_code}]: {resp.text}")
        session_id = resp.json()["session_id"]

        cursor = 0
        events: list[dict] = []
        deadline = time.time() + _POLL_TIMEOUT_S
        status = "running"

        while time.time() < deadline:
            r = client.get(f"/api/status/{session_id}", params={"cursor": cursor})
            data = r.json()
            status = data["status"]
            events.extend(data.get("events", []))
            cursor = data.get("cursor", cursor)

            if status == "error":
                err = next(
                    (e.get("message") for e in reversed(events) if e.get("type") == "error"),
                    "unknown error",
                )
                raise RuntimeError(f"Agent run failed: {err}")

            if status in ("awaiting_feedback", "completed"):
                break

            time.sleep(_POLL_INTERVAL_S)
        else:
            raise RuntimeError(
                f"Agent did not reach review within {_POLL_TIMEOUT_S}s (last status={status})."
            )

        outputs = _latest_state(events)
        if not outputs:
            raise RuntimeError("Reached review but no output state was captured.")

        # Approve to let the background thread finish and free the session.
        if approve and status == "awaiting_feedback":
            client.post(f"/api/feedback/{session_id}", json={"approve": True})

        return outputs
