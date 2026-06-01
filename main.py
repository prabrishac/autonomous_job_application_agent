"""
main.py — CLI runner for the Autonomous Job Application Agent.

Usage:
    cd /Users/prabrisha/agentic_ai/aiengg/capstone_project
    python -m autonomous_job_application_agent.main
"""

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from autonomous_job_application_agent.graph import build_graph
from autonomous_job_application_agent.state import AgentState


SAMPLE_JD = """
Software Engineer, Backend — Stripe

About Stripe:
Stripe is a technology company that builds economic infrastructure for the internet.
Businesses of every size use our software and APIs to accept payments online and in
person, send payouts, and manage their businesses online.

What you'll do:
- Design and build reliable, scalable APIs serving millions of requests per day
- Work closely with product and infrastructure teams to ship new payment features
- Improve developer experience by making our APIs clear, consistent, and performant
- Debug and resolve production incidents with a focus on root cause elimination
- Contribute to technical roadmap and architecture decisions

We're looking for someone who:
- Has 3+ years of backend engineering experience (Python, Go, Java, or Ruby)
- Understands distributed systems, databases, and API design deeply
- Has strong debugging skills and a bias toward fixing root causes
- Communicates clearly in writing and code reviews
- Cares about developer experience as a product

Nice to have:
- Experience with payment systems or financial infrastructure
- Open source contributions
- Experience scaling services to millions of users
"""

SAMPLE_RESUME = """
Jane Smith
jane@example.com | github.com/janesmith | San Francisco, CA

SUMMARY
Backend engineer with 4 years of experience building Python and Go services.
Passionate about clean API design and distributed systems.

EXPERIENCE

Senior Software Engineer — Acme Corp (2022–present)
- Built a high-throughput event processing pipeline handling 50K events/sec using Python + Kafka
- Led redesign of public REST API reducing average latency by 40%
- Mentored 2 junior engineers; introduced code review standards across the team
- Reduced production incidents 30% by improving observability with Datadog and structured logging

Software Engineer — StartupXYZ (2020–2022)
- Developed core billing microservice in Go, integrated with Stripe and Braintree
- Designed PostgreSQL schema for subscription lifecycle management
- Contributed to open-source gRPC middleware library (800+ GitHub stars)
- Worked in a fast-paced environment shipping weekly to production

SKILLS
Languages: Python, Go, TypeScript
Databases: PostgreSQL, Redis, DynamoDB
Infrastructure: AWS, Docker, Kubernetes, Terraform
Tools: Kafka, gRPC, REST APIs, Datadog, GitHub Actions

EDUCATION
B.Sc. Computer Science — UC Berkeley, 2020
"""


def print_separator(title: str = ""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * width)


def display_state(state: dict):
    import json
    print_separator("COMPANY RESEARCH")
    print(state.get("company_research", "—")[:600])

    print_separator("JD ANALYSIS")
    print(json.dumps(state.get("jd_analysis", {}), indent=2)[:800])

    print_separator("TAILORED RESUME (excerpt)")
    print(state.get("tailored_resume", "—")[:800])

    print_separator("COVER LETTER")
    print(state.get("cover_letter", "—"))

    print_separator("INTERVIEW PREP (excerpt)")
    print("\n".join(state.get("interview_questions", [])[:20]))

    print_separator("QUALITY CHECK")
    print(f"Score: {state.get('quality_score', 0):.2f}")
    print(state.get("quality_feedback", ""))


def export_outputs(state: dict):
    out_dir = Path("/Users/prabrisha/agentic_ai/aiengg/capstone_project/autonomous_job_application_agent/outputs")
    out_dir.mkdir(exist_ok=True)

    company = state.get("company_name", "company").replace(" ", "_").lower()

    files = {
        f"{company}_resume.md": state.get("tailored_resume", ""),
        f"{company}_cover_letter.md": state.get("cover_letter", ""),
        f"{company}_interview_prep.md": "\n".join(state.get("interview_questions", [])),
        f"{company}_company_research.md": state.get("company_research", ""),
    }

    for filename, content in files.items():
        path = out_dir / filename
        path.write_text(content)
        print(f"  📁 Saved: {path}")

    print(f"\nAll files exported to {out_dir}")


def main():
    print("═" * 60)
    print("   AUTONOMOUS JOB APPLICATION AGENT")
    print("   Powered by LangGraph + GPT-4o")
    print("═" * 60)

    # Load inputs — swap with file reads if needed:
    # jd = Path("job_description.txt").read_text()
    # resume = Path("my_resume.txt").read_text()
    jd = SAMPLE_JD
    resume = SAMPLE_RESUME

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "job_description": jd,
        "resume_text": resume,
        "company_name": "",
        "company_research": "",
        "jd_analysis": {},
        "tailored_resume": "",
        "cover_letter": "",
        "interview_questions": [],
        "quality_score": 0.0,
        "quality_feedback": "",
        "revision_count": 0,
        "human_feedback": "",
        "messages": [],
    }

    print("\n🚀 Starting agent pipeline...\n")

    for event in graph.stream(initial_state, config=config):
        for node_name, updates in event.items():
            if node_name == "__interrupt__":
                continue
            for msg in updates.get("messages", []):
                print(f"  {msg}")

    current = graph.get_state(config)
    state_values = current.values

    while True:
        display_state(state_values)

        print_separator("HUMAN REVIEW")
        print(f"Quality score: {state_values.get('quality_score', 0):.2f}")
        print(f"Revision #{state_values.get('revision_count', 0)}")
        print("\nOptions:")
        print("  • Type 'approve' to finalise and export documents")
        print("  • Type feedback to improve specific sections")
        print("  • Examples: 'make cover letter shorter', 'add more Python to resume'\n")

        feedback = input("Your input: ").strip()
        if not feedback:
            feedback = "approve"

        graph.update_state(config, {"human_feedback": feedback})

        if feedback.lower() in ("approve", "approved", "ok", "looks good", "yes"):
            for event in graph.stream(None, config=config):
                pass
            print("\n✅ Approved! Exporting documents...")
            export_outputs(state_values)
            break
        else:
            print("\n🔄 Regenerating with your feedback...\n")
            for event in graph.stream(None, config=config):
                for node_name, updates in event.items():
                    if node_name == "__interrupt__":
                        continue
                    for msg in updates.get("messages", []):
                        print(f"  {msg}")

            current = graph.get_state(config)
            state_values = current.values


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("  Add it to the .env file in the project directory:")
        print("  OPENAI_API_KEY=sk-...")
        sys.exit(1)
    main()
