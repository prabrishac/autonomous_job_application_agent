"""
graph.py — Builds and compiles the LangGraph StateGraph.

Topology:
  START → parse_input → research_company → analyze_jd
        → [tailor_resume, write_cover_letter, prepare_interview]   ← parallel
        → aggregate_outputs
        → human_review | retry loop                                ← conditional
        → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from autonomous_job_application_agent.state import AgentState
from autonomous_job_application_agent.agents.nodes import (
    parse_input,
    research_company,
    analyze_jd,
    tailor_resume,
    write_cover_letter,
    prepare_interview,
    aggregate_outputs,
    human_review,
)


def route_after_aggregate(state: AgentState) -> str:
    score = state.get("quality_score", 0.0)
    revisions = state.get("revision_count", 0)
    if score >= 0.7 or revisions >= 3:
        return "human_review"
    return "tailor_resume"


def route_after_human(state: AgentState) -> str:
    feedback = state.get("human_feedback", "").strip().lower()
    if feedback in ("approve", "approved", "ok", "looks good", "yes"):
        return END
    return "tailor_resume"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("parse_input", parse_input)
    builder.add_node("research_company", research_company)
    builder.add_node("analyze_jd", analyze_jd)
    builder.add_node("tailor_resume", tailor_resume)
    builder.add_node("write_cover_letter", write_cover_letter)
    builder.add_node("prepare_interview", prepare_interview)
    builder.add_node("aggregate_outputs", aggregate_outputs)
    builder.add_node("human_review", human_review)

    # Sequential pipeline
    builder.add_edge(START, "parse_input")
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
    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {END: END, "tailor_resume": "tailor_resume"}
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["human_review"])
