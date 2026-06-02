"""
AgentState — the single shared state object passed through every LangGraph node.
All agents read from and write to this TypedDict.
"""

from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    job_description: str          # Raw JD text pasted by user
    resume_text: str              # Raw resume text pasted by user
    company_name: str             # Extracted or supplied company name

    # ── PII guardrail ─────────────────────────────────────────────────────────
    pii_report: dict              # {pii_type: count} — safe to log, no raw values
    pii_mapping: dict             # {token: original_value} — used to restore PII in outputs
    sanitized_resume: str         # Resume with PII replaced by tokens (used by analysis nodes)

    # ── Research outputs ──────────────────────────────────────────────────────
    company_research: str         # Summary from web search about company/role
    jd_analysis: dict             # Structured: {skills, keywords, tone, requirements}

    # ── Generated documents ───────────────────────────────────────────────────
    tailored_resume: str          # Rewritten resume sections optimised for this JD
    cover_letter: str             # Full cover letter draft
    interview_questions: list[str]  # Predicted interview Q&A pairs

    # ── Quality & control ─────────────────────────────────────────────────────
    quality_score: float          # 0–1 score from aggregator
    quality_feedback: str         # Human-readable gaps / suggestions
    revision_count: int           # How many retry loops have run
    human_feedback: str           # Combined feedback string (used by classify_feedback)
    feedback_type: str            # "approve" | "actionable" | "irrelevant" — set by classify_feedback

    # ── Per-document feedback (set by UI, consumed by generation nodes) ────────
    resume_feedback: str          # Specific feedback for the tailored resume
    cover_letter_feedback: str    # Specific feedback for the cover letter
    interview_feedback: str       # Specific feedback for the interview prep guide

    # ── Accumulate log messages across all nodes ───────────────────────────────
    messages: Annotated[list[str], operator.add]
