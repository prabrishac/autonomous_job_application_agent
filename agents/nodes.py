"""
Agent nodes for the Job Application LangGraph.

Each function takes AgentState and returns a dict of state updates.
LangGraph merges the returned dict back into the shared state automatically.

Node execution order (see graph.py):
  parse_input → research_company → analyze_jd
      → [resume_tailor, cover_letter_writer, interview_prep]  (parallel)
      → aggregate_outputs → human_review
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from autonomous_job_application_agent.state import AgentState
from autonomous_job_application_agent.tools.search_tools import web_search, fetch_url
from autonomous_job_application_agent.guardrails.pii_guard import (
    scan_pii,
    redact_pii,
    restore_pii,
    strip_stray_pii_tokens,
    sanitize_log,
    assert_no_sensitive_pii,
    PIIBlockedError,
)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 0 — PII guardrail  (runs first, before any LLM call)
#
# What it does:
#   1. Scans resume_text for PII using regex patterns.
#   2. Blocks immediately if sensitive-tier PII (SSN, credit card, passport)
#      is found — raises PIIBlockedError before anything leaves the device.
#   3. Redacts standard PII (email, phone, address, URLs) with reversible
#      [PII:TYPE:uid] tokens and stores the mapping in state.
#   4. Saves the cleaned copy as sanitized_resume — analysis nodes (research,
#      analyze_jd) use this so raw contact details never reach the LLM.
#   5. tailor_resume and write_cover_letter use the original resume_text
#      because the contact info must appear in the final documents; the PII
#      mapping is used to restore tokens in those outputs if needed.
# ─────────────────────────────────────────────────────────────────────────────

def pii_guardrail(state: AgentState) -> dict:
    """
    Gate-keeper node — must be the first node in the graph.
    Scans inputs for PII and populates state with the redacted copy + mapping.
    """
    resume = state["resume_text"]

    # ── Step 1: detect ────────────────────────────────────────────────────────
    report, findings = scan_pii(resume)

    # ── Step 2: block on sensitive PII ────────────────────────────────────────
    try:
        assert_no_sensitive_pii(report, source="resume")
    except PIIBlockedError as exc:
        raise PIIBlockedError(str(exc)) from exc

    # ── Step 3: redact standard PII for analysis nodes ────────────────────────
    sanitized, mapping = redact_pii(resume, findings)

    # ── Step 4: build a safe summary for logs (no raw values) ─────────────────
    if report.counts:
        summary_parts = [f"{t}×{n}" for t, n in sorted(report.counts.items())]
        log_msg = "🔒 PII guardrail — detected in resume: " + ", ".join(summary_parts) + \
                  ". Redacted for LLM analysis nodes; restored in final documents."
    else:
        log_msg = "🔒 PII guardrail — no PII detected in resume."

    return {
        "pii_report": report.counts,
        "pii_mapping": mapping,
        "sanitized_resume": sanitized,
        "messages": [log_msg],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM instance
# Use gpt-4o for best results; gpt-4o-mini works fine for cheaper runs.
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.3):
    return ChatOpenAI(model="gpt-4o", temperature=temperature)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — Input parser
# Extracts company name and validates inputs before any LLM calls happen.
# ─────────────────────────────────────────────────────────────────────────────

def parse_input(state: AgentState) -> dict:
    """
    Validates inputs and extracts the company name from the job description.
    No LLM call needed — simple extraction to avoid wasting tokens.
    """
    jd = state["job_description"]
    resume = state["resume_text"]

    if not jd.strip():
        raise ValueError("Job description cannot be empty.")
    if not resume.strip():
        raise ValueError("Resume cannot be empty.")

    # Try to extract company name from JD using a cheap LLM call
    llm = get_llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content="Extract only the company name from this job description. "
                               "Return just the company name, nothing else."),
        HumanMessage(content=jd[:2000])
    ])
    company_name = response.content.strip()

    return {
        "company_name": company_name,
        "revision_count": state.get("revision_count", 0),
        "messages": [f"✅ Input parsed. Company identified: {company_name}"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — Company researcher
# Uses web_search tool to gather intel on the company and role.
# ─────────────────────────────────────────────────────────────────────────────

def research_company(state: AgentState) -> dict:
    """
    Searches the web for company background, culture, recent news, and role context.
    Uses the web_search tool — demonstrates tool-calling pattern.
    """
    company = state["company_name"]

    # Build search queries
    queries = [
        f"{company} company culture values mission",
        f"{company} recent news 2024 2025",
        f"{company} engineering team tech stack",
    ]

    research_chunks = []
    for q in queries:
        result = web_search.invoke({"query": q})
        research_chunks.append(f"### {q}\n{result}")

    # Ask LLM to synthesise the raw search results into a useful brief
    llm = get_llm(temperature=0.2)
    synthesis = llm.invoke([
        SystemMessage(content=(
            "You are a career research assistant. Synthesise the web search results "
            "below into a concise company brief (max 400 words). Include: company mission, "
            "culture signals, tech stack if mentioned, recent news, and what they seem to "
            "value in employees. Be factual and direct."
        )),
        HumanMessage(content="\n\n".join(research_chunks))
    ])

    return {
        "company_research": synthesis.content,
        "messages": [f"🔍 Company research complete for {company}."]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — JD analyzer
# Structured extraction of skills, keywords, tone, and requirements.
# ─────────────────────────────────────────────────────────────────────────────

def analyze_jd(state: AgentState) -> dict:
    """
    Extracts structured information from the job description:
    required skills, nice-to-haves, keywords for ATS, seniority level, and tone.
    Returns a dict stored in state["jd_analysis"].
    """
    llm = get_llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content=(
            "Analyze this job description and return a JSON object with these keys:\n"
            "- required_skills: list of must-have technical/soft skills\n"
            "- nice_to_have: list of optional skills\n"
            "- ats_keywords: list of exact phrases to include for ATS matching\n"
            "- seniority: one of [junior, mid, senior, staff, lead, manager]\n"
            "- tone: one of [formal, conversational, startup, corporate]\n"
            "- key_responsibilities: list of 3-5 main job duties\n"
            "- company_values_signals: list of values/culture signals mentioned\n"
            "Return ONLY valid JSON, no markdown fences."
        )),
        HumanMessage(content=state["job_description"])
    ])

    try:
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```")
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        analysis = {"raw_analysis": response.content}

    return {
        "jd_analysis": analysis,
        "messages": ["📋 JD analysis complete. Skills and keywords extracted."]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4a — Resume tailor
# ─────────────────────────────────────────────────────────────────────────────

def tailor_resume(state: AgentState) -> dict:
    """
    Rewrites the candidate's resume to match the JD — no fabrication.
    Skips regeneration when no resume-specific feedback is provided and content already exists.
    """
    resume_fb = state.get("resume_feedback", "").strip()
    existing  = state.get("tailored_resume", "").strip()

    if not resume_fb and existing:
        return {"messages": ["📄 Resume unchanged (no specific feedback)."]}

    llm = get_llm(temperature=0.4)

    feedback = resume_fb or state.get("human_feedback", "")
    feedback_section = (
        f"\n\nFeedback to incorporate:\n{feedback}"
        if feedback else ""
    )

    # Use sanitized_resume (PII redacted) for the LLM call; restore real
    # contact details in the output so the final document is complete.
    sanitized = state.get("sanitized_resume") or state["resume_text"]
    mapping = state.get("pii_mapping", {})

    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert resume writer and career coach. "
            "Rewrite the candidate's resume to be a strong match for the target role "
            "without inventing any experience.\n"
            "Rules:\n"
            "1. Mirror exact ATS keywords from the JD naturally.\n"
            "2. Rewrite bullet points to highlight relevant impact.\n"
            "3. Rewrite the summary/objective for this role.\n"
            "4. Do NOT fabricate experience, companies, or skills.\n"
            "5. Output in clean markdown. Do NOT wrap the response in a code fence (no ```).\n"
            "6. Preserve all real experience — only reframe it.\n"
            "7. The resume text may contain tokens like [PII:PHONE:abc12345]. "
            "Copy those tokens exactly as they appear — do NOT invent new [PII:*] tokens "
            "and do NOT use [PII:TYPE] shorthand without the hex ID. "
            "The tokens will be replaced with real contact details after your response."
        )),
        HumanMessage(content=(
            f"## Original Resume\n{sanitized}\n\n"
            f"## Job Description\n{state['job_description']}\n\n"
            f"## Key Skills & Keywords to Include\n"
            f"{json.dumps(state.get('jd_analysis', {}), indent=2)}\n\n"
            f"## Company Research\n{state.get('company_research', '')}"
            f"{feedback_section}"
        ))
    ])

    final_resume = strip_stray_pii_tokens(restore_pii(response.content, mapping))

    return {
        "tailored_resume": final_resume,
        "resume_feedback": "",  # clear after use
        "messages": ["📄 Resume tailored for the role."]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4b — Cover letter writer
# ─────────────────────────────────────────────────────────────────────────────

def write_cover_letter(state: AgentState) -> dict:
    """
    Generates a personalised, tone-matched cover letter.
    Skips regeneration when no cover-letter-specific feedback is provided and content already exists.
    """
    cover_fb = state.get("cover_letter_feedback", "").strip()
    existing = state.get("cover_letter", "").strip()

    if not cover_fb and existing:
        return {"messages": ["✉️ Cover letter unchanged (no specific feedback)."]}

    llm = get_llm(temperature=0.6)

    tone    = state.get("jd_analysis", {}).get("tone", "professional")
    company = state["company_name"]

    feedback = cover_fb or state.get("human_feedback", "")
    feedback_section = (
        f"\n\nFeedback to incorporate:\n{feedback}"
        if feedback else ""
    )

    # Use sanitized_resume (PII redacted) for the LLM call; restore real
    # contact details in the output so the final document is complete.
    sanitized = state.get("sanitized_resume") or state["resume_text"]
    mapping = state.get("pii_mapping", {})

    response = llm.invoke([
        SystemMessage(content=(
            f"You are an expert cover letter writer. Write in a {tone} tone.\n"
            "Rules:\n"
            "1. Opening: specific hook — mention something real about the company.\n"
            "2. Body paragraph 1: connect 2 key experiences to key responsibilities.\n"
            "3. Body paragraph 2: demonstrate culture/values alignment.\n"
            "4. Closing: confident, specific call to action.\n"
            "5. Total length: 3-4 paragraphs, under 350 words.\n"
            "6. NO clichés: avoid 'I am writing to express', 'passion for', 'team player'.\n"
            "7. Output in clean markdown. Do NOT wrap the response in a code fence (no ```).\n"
            "8. The resume text may contain tokens like [PII:PHONE:abc12345]. "
            "Copy those tokens exactly as they appear — do NOT invent new [PII:*] tokens "
            "and do NOT use [PII:TYPE] shorthand without the hex ID. "
            "The tokens will be replaced with real contact details after your response."
        )),
        HumanMessage(content=(
            f"## Target Company: {company}\n\n"
            f"## Company Research\n{state.get('company_research', '')}\n\n"
            f"## Job Description\n{state['job_description']}\n\n"
            f"## Candidate Resume\n{sanitized}\n\n"
            f"## JD Analysis\n{json.dumps(state.get('jd_analysis', {}), indent=2)}"
            f"{feedback_section}"
        ))
    ])

    final_letter = strip_stray_pii_tokens(restore_pii(response.content, mapping))

    return {
        "cover_letter": final_letter,
        "cover_letter_feedback": "",  # clear after use
        "messages": ["✉️ Cover letter written."]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4c — Interview prep
# ─────────────────────────────────────────────────────────────────────────────

def prepare_interview(state: AgentState) -> dict:
    """
    Generates technical questions, STAR behavioural Q&As, and questions to ask.
    Skips regeneration when no interview-specific feedback is provided and content already exists.
    """
    interview_fb = state.get("interview_feedback", "").strip()
    existing     = state.get("interview_questions", [])

    if not interview_fb and existing:
        return {"messages": ["🎯 Interview prep unchanged (no specific feedback)."]}

    llm = get_llm(temperature=0.4)
    company = state["company_name"]

    sanitized = state.get("sanitized_resume") or state["resume_text"]

    feedback_section = (
        f"\n\n## Feedback to incorporate\n{interview_fb}"
        if interview_fb else ""
    )

    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior interview coach. Generate a targeted interview prep guide.\n"
            "Structure:\n"
            "## Technical Questions (5 questions with brief model answers)\n"
            "## Behavioural Questions (5 STAR-format Q&As based on the candidate's real experience)\n"
            "## Culture-Fit Questions (3 questions linking to company values)\n"
            "## Questions to Ask the Interviewer (5 smart, specific questions)\n\n"
            "Base STAR answers ONLY on the candidate's actual resume — no fabrication."
        )),
        HumanMessage(content=(
            f"## Company: {company}\n\n"
            f"## Company Research\n{state.get('company_research', '')}\n\n"
            f"## Job Description\n{state['job_description']}\n\n"
            f"## Candidate Resume\n{sanitized}\n\n"
            f"## JD Analysis\n{json.dumps(state.get('jd_analysis', {}), indent=2)}"
            f"{feedback_section}"
        ))
    ])

    lines = [l for l in response.content.split("\n") if l.strip()]

    return {
        "interview_questions": lines,
        "interview_feedback": "",  # clear after use
        "messages": ["🎯 Interview prep guide generated."]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — Aggregator + quality check
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_outputs(state: AgentState) -> dict:
    """
    Scores all outputs 0.0–1.0 and surfaces specific feedback.
    """
    llm = get_llm(temperature=0)

    response = llm.invoke([
        SystemMessage(content=(
            "You are a quality-control agent reviewing job application materials.\n"
            "Evaluate the resume, cover letter, and interview prep against the JD and return "
            "a JSON object with:\n"
            "- score: float 0.0 to 1.0 (overall quality)\n"
            "- resume_feedback: specific gaps in the tailored resume\n"
            "- cover_letter_feedback: specific issues with the cover letter\n"
            "- overall_feedback: 2-3 sentence summary for the candidate\n"
            "Scoring: 0.9+ = publish-ready, 0.7-0.9 = minor tweaks, <0.7 = needs rework.\n"
            "Return ONLY valid JSON."
        )),
        HumanMessage(content=(
            f"## Job Description\n{state['job_description']}\n\n"
            f"## Tailored Resume\n{state.get('tailored_resume', '')}\n\n"
            f"## Cover Letter\n{state.get('cover_letter', '')}\n\n"
            f"## Interview Prep (first 500 chars)\n"
            f"{chr(10).join(state.get('interview_questions', []))[:500]}"
        ))
    ])

    try:
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```")
        result = json.loads(raw)
        score = float(result.get("score", 0.5))
        feedback = result.get("overall_feedback", "Review complete.")
    except Exception:
        score = 0.5
        feedback = response.content[:300]

    # Sanitize feedback before storing in logs — quality feedback should never
    # echo back PII from the resume into the message stream.
    safe_feedback = sanitize_log(feedback)

    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "messages": [f"🔎 Quality check: score={score:.2f}. {safe_feedback}"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — Human review (HITL interrupt node)
# ─────────────────────────────────────────────────────────────────────────────

def human_review(state: AgentState) -> dict:
    """
    Interrupt point — LangGraph pauses here awaiting human input.
    Feedback is injected via graph.update_state() in main.py then graph is resumed.
    revision_count is NOT incremented here — only classify_feedback does that,
    and only when feedback is actionable (prevents count inflation on irrelevant input).
    """
    feedback = state.get("human_feedback", "approve")
    return {
        "messages": [f"👤 Human review: '{feedback[:80]}'"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7 — Feedback classifier
#
# Runs immediately after the human_review interrupt is resumed.
# Uses a fast LLM call to put the feedback into one of three buckets:
#
#   approve    → user is satisfied; graph routes to END
#   actionable → feedback contains specific, usable improvement instructions;
#                graph routes to tailor_resume for regeneration
#   irrelevant → feedback is gibberish, too vague, or off-topic;
#                graph routes back to human_review (triggers interrupt again)
#                so the user is prompted without wasting an LLM cycle
# ─────────────────────────────────────────────────────────────────────────────

_APPROVAL_KEYWORDS = frozenset(
    ["approve", "approved", "ok", "looks good", "yes", "lgtm", "ship it", "y", "done"]
)

def classify_feedback(state: AgentState) -> dict:
    """
    Classify human_feedback and set feedback_type in state.
    Increments revision_count only on actionable feedback.
    """
    feedback = state.get("human_feedback", "").strip()

    # ── Fast path: explicit approval keyword ──────────────────────────────────
    if feedback.lower() in _APPROVAL_KEYWORDS:
        return {
            "feedback_type": "approve",
            "messages": ["✅ Feedback classified as approval."],
        }

    # ── Fast path: empty or too short to be useful ────────────────────────────
    if len(feedback) < 8:
        return {
            "feedback_type": "irrelevant",
            "messages": [
                f"⚠️  Feedback '{feedback}' is too short to act on. "
                "Please describe what to improve (e.g. 'make the summary shorter') "
                "or type 'approve' to finish."
            ],
        }

    # ── LLM classification ────────────────────────────────────────────────────
    llm = get_llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a feedback classifier for a job application assistant.\n"
            "The assistant produces a tailored resume, cover letter, and interview prep guide.\n\n"
            "Classify the user's feedback into exactly one of these categories:\n"
            "  approve    — user is satisfied and wants to finish "
            "(e.g. 'looks good', 'all good', 'perfect', 'send it')\n"
            "  actionable — feedback contains specific, usable instructions for improving "
            "the resume, cover letter, or interview prep "
            "(e.g. 'add more Python keywords', 'shorten the cover letter', "
            "'focus on leadership experience')\n"
            "  irrelevant — feedback is gibberish, random characters, off-topic, "
            "or too vague to act on "
            "(e.g. 'asdf', 'I don't know', 'whatever', 'hello', general questions)\n\n"
            "Reply with ONLY one word: approve, actionable, or irrelevant."
        )),
        HumanMessage(content=f"User feedback: {feedback}")
    ])

    classification = response.content.strip().lower()
    if classification not in ("approve", "actionable", "irrelevant"):
        classification = "irrelevant"

    if classification == "approve":
        return {
            "feedback_type": "approve",
            "messages": ["✅ Feedback classified as approval."],
        }

    if classification == "actionable":
        return {
            "feedback_type": "actionable",
            "revision_count": state.get("revision_count", 0) + 1,
            "messages": [f"✏️  Actionable feedback — regenerating documents with: '{feedback[:80]}'"],
        }

    # irrelevant
    return {
        "feedback_type": "irrelevant",
        "messages": [
            f"⚠️  Feedback doesn't seem actionable: '{feedback[:60]}'. "
            "Please provide specific improvement instructions or type 'approve' to finish."
        ],
    }
