"""
api.py — FastAPI server for the Autonomous Job Application Agent UI.

Run from the project parent directory:
    cd /Users/prabrisha/agentic_ai/aiengg/capstone_project
    uvicorn autonomous_job_application_agent.api:app --reload --port 8000
"""

import os
import sys
import uuid
import threading
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from autonomous_job_application_agent.graph import build_graph
from autonomous_job_application_agent.state import AgentState
from autonomous_job_application_agent.guardrails.pii_guard import PIIBlockedError
from autonomous_job_application_agent.utils.pdf_export import build_pdf
from autonomous_job_application_agent.utils.pdf_parser import extract_pdf_text

app = FastAPI(title="Autonomous Job Application Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# In-memory session store  {session_id: session_dict}
sessions: dict = {}


# ── Pydantic models ────────────────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    job_description: str
    resume_text: str


class FeedbackRequest(BaseModel):
    resume_feedback: str = ""
    cover_letter_feedback: str = ""
    interview_feedback: str = ""
    approve: bool = False


class ExportPDFRequest(BaseModel):
    title: str
    content: str
    filename: str = "document.pdf"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _push(session: dict, event: dict) -> None:
    session["events"].append(event)


def _output_payload(state: dict) -> dict:
    qs = state.get("interview_questions", [])
    if isinstance(qs, list):
        interview_text = "\n\n".join(str(q) for q in qs)
    else:
        interview_text = str(qs)
    return {
        "tailored_resume": state.get("tailored_resume", ""),
        "cover_letter": state.get("cover_letter", ""),
        "interview_questions": interview_text,
        "quality_score": state.get("quality_score", 0.0),
        "quality_feedback": state.get("quality_feedback", ""),
        "company_name": state.get("company_name", ""),
        "revision_count": state.get("revision_count", 0),
        "feedback_type": state.get("feedback_type", ""),
    }


# ── Background agent thread ────────────────────────────────────────────────────

def _run_agent(session_id: str, jd: str, resume_text: str) -> None:
    session = sessions[session_id]

    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to the .env file.")

        graph = build_graph()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        initial_state: AgentState = {
            "job_description": jd,
            "resume_text": resume_text,
            "company_name": "",
            "pii_report": {},
            "pii_mapping": {},
            "sanitized_resume": "",
            "company_research": "",
            "jd_analysis": {},
            "tailored_resume": "",
            "cover_letter": "",
            "interview_questions": [],
            "tailored_resume_sanitized": "",
            "cover_letter_sanitized": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "revision_count": 0,
            "human_feedback": "",
            "feedback_type": "",
            "approved": False,
            "resume_feedback": "",
            "cover_letter_feedback": "",
            "interview_feedback": "",
            "messages": [],
        }

        _push(session, {"type": "started"})
        print(f"\n[{session_id[:8]}] 🚀 Agent started")

        # ── Initial run ───────────────────────────────────────────────────────
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, updates in event.items():
                if node_name == "__interrupt__":
                    continue
                print(f"[{session_id[:8]}] ✓ node: {node_name}")
                for msg in updates.get("messages", []):
                    print(f"[{session_id[:8]}]    {msg}")
                _push(session, {"type": "node_complete", "node": node_name})

        current = graph.get_state(config)
        snapshot = dict(current.values)

        # ── HITL loop ─────────────────────────────────────────────────────────
        while current.next:
            score = snapshot.get("quality_score", 0.0)
            rev = snapshot.get("revision_count", 0)
            print(f"[{session_id[:8]}] ⏸  awaiting feedback (score={score:.2f}, revision={rev})")
            _push(session, {"type": "awaiting_feedback", "state": _output_payload(snapshot)})
            session["status"] = "awaiting_feedback"

            session["feedback_ready"].wait()
            session["feedback_ready"].clear()
            session["status"] = "running"
            print(f"[{session_id[:8]}] ▶  feedback received, resuming...")

            fd = session["pending_feedback"]
            is_approved = bool(fd.get("approve"))

            if is_approved:
                # Explicit approval — finalise via the approved flag, NOT by
                # passing "approve" as feedback text (which classify_feedback
                # could misread on a per-document message).
                rf, cf, itf = "", "", ""
                combined = "approve"
            else:
                rf  = fd.get("resume_feedback", "").strip()
                cf  = fd.get("cover_letter_feedback", "").strip()
                itf = fd.get("interview_feedback", "").strip()
                parts = []
                if rf:  parts.append(f"Resume: {rf}")
                if cf:  parts.append(f"Cover Letter: {cf}")
                if itf: parts.append(f"Interview Prep: {itf}")
                combined = "; ".join(parts)   # may be "" — handled as irrelevant

            print(f"[{session_id[:8]}]    approve={is_approved} feedback: {combined[:120]}")
            graph.update_state(config, {
                "approved": is_approved,
                "human_feedback": combined,
                "resume_feedback": rf,
                "cover_letter_feedback": cf,
                "interview_feedback": itf,
            })

            for event in graph.stream(None, config=config, stream_mode="updates"):
                for node_name, updates in event.items():
                    if node_name == "__interrupt__":
                        continue
                    print(f"[{session_id[:8]}] ✓ node: {node_name}")
                    for msg in updates.get("messages", []):
                        print(f"[{session_id[:8]}]    {msg}")
                    _push(session, {"type": "node_complete", "node": node_name})

            current = graph.get_state(config)
            snapshot = dict(current.values)

        # ── Completed ─────────────────────────────────────────────────────────
        print(f"[{session_id[:8]}] ✅ Completed — exporting documents")
        _push(session, {"type": "completed", "state": _output_payload(snapshot)})
        session["status"] = "completed"

    except PIIBlockedError as exc:
        print(f"[{session_id[:8]}] 🚫 BLOCKED — PII detected: {exc}")
        _push(session, {"type": "error", "message": f"Sensitive PII detected in resume: {exc}"})
        session["status"] = "error"
    except Exception as exc:
        print(f"[{session_id[:8]}] ❌ ERROR — {exc}")
        _push(session, {"type": "error", "message": str(exc), "detail": traceback.format_exc()})
        session["status"] = "error"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(_static_dir / "index.html"))


@app.post("/api/submit")
async def submit(req: SubmitRequest):
    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is required.")
    if not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text is required.")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "status": "running",
        "events": [],
        "feedback_ready": threading.Event(),
        "pending_feedback": None,
    }

    t = threading.Thread(
        target=_run_agent,
        args=(session_id, req.job_description, req.resume_text),
        daemon=True,
    )
    t.start()

    return {"session_id": session_id}


@app.get("/api/status/{session_id}")
async def get_status(session_id: str, cursor: int = 0):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    new_events = session["events"][cursor:]

    return {
        "status": session["status"],
        "events": new_events,
        "cursor": len(session["events"]),
    }


@app.post("/api/feedback/{session_id}")
async def submit_feedback(session_id: str, req: FeedbackRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    if session["status"] != "awaiting_feedback":
        raise HTTPException(status_code=400, detail="Agent is not currently waiting for feedback.")

    session["pending_feedback"] = req.model_dump()
    session["feedback_ready"].set()

    return {"status": "feedback_received"}


@app.post("/api/export-pdf")
async def export_pdf(req: ExportPDFRequest):
    """
    Generate a PDF from markdown *content* and stream it back for download.
    The client already holds the document text, so no session look-up is needed.
    """
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content is empty.")

    pdf_bytes = build_pdf(req.title, req.content)

    safe_name = req.filename.replace('"', "").replace("/", "_") or "document.pdf"
    if not safe_name.endswith(".pdf"):
        safe_name += ".pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload and return the extracted plain text.

    The client uses this to populate the Job Description / Resume textareas
    without a page reload.  Only PDF content type is accepted.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    data = await file.read()

    try:
        text, page_count = extract_pdf_text(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "text": text,
        "pages": page_count,
        "filename": filename,
        "chars": len(text),
    }
