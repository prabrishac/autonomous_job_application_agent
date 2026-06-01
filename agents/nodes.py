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
    """
    llm = get_llm(temperature=0.4)

    human_feedback = state.get("human_feedback", "")
    feedback_section = (
        f"\n\nPrevious feedback to incorporate:\n{human_feedback}"
        if human_feedback else ""
    )

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
            "5. Output in clean markdown format.\n"
            "6. Preserve all real experience — only reframe it."
        )),
        HumanMessage(content=(
            f"## Original Resume\n{state['resume_text']}\n\n"
            f"## Job Description\n{state['job_description']}\n\n"
            f"## Key Skills & Keywords to Include\n"
            f"{json.dumps(state.get('jd_analysis', {}), indent=2)}\n\n"
            f"## Company Research\n{state.get('company_research', '')}"
            f"{feedback_section}"
        ))
    ])

    return {
        "tailored_resume": response.content,
        "messages": ["📄 Resume tailored for the role."]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4b — Cover letter writer
# ─────────────────────────────────────────────────────────────────────────────

def write_cover_letter(state: AgentState) -> dict:
    """
    Generates a personalised, tone-matched cover letter.
    """
    llm = get_llm(temperature=0.6)

    tone = state.get("jd_analysis", {}).get("tone", "professional")
    company = state["company_name"]

    human_feedback = state.get("human_feedback", "")
    feedback_section = (
        f"\n\nPrevious feedback to incorporate:\n{human_feedback}"
        if human_feedback else ""
    )

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
            "7. Output in clean markdown."
        )),
        HumanMessage(content=(
            f"## Target Company: {company}\n\n"
            f"## Company Research\n{state.get('company_research', '')}\n\n"
            f"## Job Description\n{state['job_description']}\n\n"
            f"## Candidate Resume\n{state['resume_text']}\n\n"
            f"## JD Analysis\n{json.dumps(state.get('jd_analysis', {}), indent=2)}"
            f"{feedback_section}"
        ))
    ])

    return {
        "cover_letter": response.content,
        "messages": ["✉️ Cover letter written."]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4c — Interview prep
# ─────────────────────────────────────────────────────────────────────────────

def prepare_interview(state: AgentState) -> dict:
    """
    Generates technical questions, STAR behavioural Q&As, and questions to ask.
    """
    llm = get_llm(temperature=0.4)
    company = state["company_name"]

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
            f"## Candidate Resume\n{state['resume_text']}\n\n"
            f"## JD Analysis\n{json.dumps(state.get('jd_analysis', {}), indent=2)}"
        ))
    ])

    lines = [l for l in response.content.split("\n") if l.strip()]

    return {
        "interview_questions": lines,
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

    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "messages": [f"🔎 Quality check: score={score:.2f}. {feedback}"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — Human review (HITL interrupt node)
# ─────────────────────────────────────────────────────────────────────────────

def human_review(state: AgentState) -> dict:
    """
    Interrupt point — LangGraph pauses here awaiting human input.
    Feedback is injected via graph.update_state() in main.py then graph is resumed.
    """
    feedback = state.get("human_feedback", "approve")

    return {
        "revision_count": state.get("revision_count", 0) + 1,
        "messages": [f"👤 Human review: '{feedback[:80]}'"]
    }
