"""
graph.py — Builds and compiles the LangGraph StateGraph.

Topology:
  START → pii_guardrail → parse_input → research_company → analyze_jd
        → [tailor_resume, write_cover_letter, prepare_interview]   ← parallel
        → aggregate_outputs
        → human_review (interrupt) → classify_feedback             ← HITL
        → approve → END
        → actionable → tailor_resume (retry loop)
        → irrelevant → human_review (re-interrupt, no regeneration)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from autonomous_job_application_agent.state import AgentState
from autonomous_job_application_agent.agents.nodes import (
    pii_guardrail,
    parse_input,
    research_company,
    analyze_jd,
    tailor_resume,
    write_cover_letter,
    prepare_interview,
    aggregate_outputs,
    human_review,
    classify_feedback,
)


def route_after_aggregate(state: AgentState) -> str:
    score = state.get("quality_score", 0.0)
    revisions = state.get("revision_count", 0)
    if score >= 0.7 or revisions >= 3:
        return "human_review"
    return "tailor_resume"


def route_after_classify(state: AgentState):
    ft = state.get("feedback_type", "irrelevant")
    if ft == "approve":
        return END
    if ft == "actionable":
        # Fan out to all three generation nodes in parallel.
        # Each node has skip logic: it regenerates only when its own feedback field is set.
        return ["tailor_resume", "write_cover_letter", "prepare_interview"]
    return "human_review"   # irrelevant → re-interrupt, no regeneration


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("pii_guardrail", pii_guardrail)
    builder.add_node("parse_input", parse_input)
    builder.add_node("research_company", research_company)
    builder.add_node("analyze_jd", analyze_jd)
    builder.add_node("tailor_resume", tailor_resume)
    builder.add_node("write_cover_letter", write_cover_letter)
    builder.add_node("prepare_interview", prepare_interview)
    builder.add_node("aggregate_outputs", aggregate_outputs)
    builder.add_node("human_review", human_review)
    builder.add_node("classify_feedback", classify_feedback)

    # Sequential pipeline
    builder.add_edge(START, "pii_guardrail")
    builder.add_edge("pii_guardrail", "parse_input")
    builder.add_edge("parse_input", "research_company")
    builder.add_edge("research_company", "analyze_jd")

    # Parallel fan-out
    builder.add_edge("analyze_jd", "tailor_resume")
    builder.add_edge("analyze_jd", "write_cover_letter")
    builder.add_edge("analyze_jd", "prepare_interview")

    # Fan-in
    builder.add_edge("tailor_resume", "aggregate_outputs")
    builder.add_edge("write_cover_letter", "aggregate_outputs")
    builder.add_edge("prepare_interview", "aggregate_outputs")

    # Conditional routing
    builder.add_conditional_edges(
        "aggregate_outputs",
        route_after_aggregate,
        {"human_review": "human_review", "tailor_resume": "tailor_resume"}
    )
    # human_review (interrupt) → classify_feedback → route based on feedback_type
    builder.add_edge("human_review", "classify_feedback")
    builder.add_conditional_edges(
        "classify_feedback",
        route_after_classify,
        {
            END: END,
            "tailor_resume": "tailor_resume",
            "write_cover_letter": "write_cover_letter",
            "prepare_interview": "prepare_interview",
            "human_review": "human_review",
        }
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["human_review"])
