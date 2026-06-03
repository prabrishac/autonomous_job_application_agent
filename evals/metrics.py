"""
metrics.py — deepeval metrics for the job-application agent.

Each metric maps to one of the agent's explicit promises:

  no_fabrication       → tailored docs invent no experience beyond the resume
                         (the system's core "no fabrication" guarantee)
  jd_alignment         → tailored resume actually targets the JD's skills/keywords
  cover_letter_quality → tone match + the prompt's "no clichés" rule
  faithfulness         → off-the-shelf grounding check vs. the resume as context

`check_no_pii_tokens` is a cheap deterministic guard (no LLM): the final
documents must never leak stray [PII:TYPE:id] tokens from the redaction layer.

The GEval judge model is gpt-4o by default; override with EVAL_MODEL.
"""

import os
import re

from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def _judge_model() -> str:
    return os.environ.get("EVAL_MODEL", "gpt-4o")


# ── LLM-as-judge metrics (GEval) ────────────────────────────────────────────────

def no_fabrication_metric() -> GEval:
    return GEval(
        name="No Fabrication",
        criteria=(
            "CONTEXT is the candidate's original resume. ACTUAL_OUTPUT is the "
            "agent-tailored resume. Judge whether the tailored resume introduces "
            "any employer, job title, degree, certification, date, metric, or "
            "skill that is NOT supported by the original resume. Reframing, "
            "rewording, and emphasising existing experience is fine and should "
            "score high. Inventing new facts must score low. Output a high score "
            "only when every concrete claim traces back to the original resume."
        ),
        evaluation_params=[
            LLMTestCaseParams.CONTEXT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.8,
        model=_judge_model(),
    )


def cover_letter_no_fabrication_metric() -> GEval:
    """Grounding judge for the cover letter: every claim must trace to the resume."""
    return GEval(
        name="Cover Letter No Fabrication",
        criteria=(
            "CONTEXT is the candidate's original resume. ACTUAL_OUTPUT is an "
            "agent-generated cover letter for that candidate. Judge whether the "
            "cover letter attributes to the candidate any employer, job title, "
            "degree, certification, date, metric, team size, or skill that is NOT "
            "supported by the original resume. Narrating, emphasising, or "
            "connecting existing experience to the role is fine and should score "
            "high. Claiming new facts about the candidate must score low. Score "
            "high only when every concrete claim about the candidate traces back "
            "to the original resume."
        ),
        evaluation_params=[
            LLMTestCaseParams.CONTEXT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.8,
        model=_judge_model(),
    )


def jd_alignment_metric() -> GEval:
    return GEval(
        name="JD Alignment",
        criteria=(
            "INPUT is a job description. ACTUAL_OUTPUT is the tailored resume. "
            "Judge how well the tailored resume targets THIS job: does it surface "
            "the relevant required skills and ATS keywords from the JD that the "
            "candidate genuinely has, and lead with the most relevant experience? "
            "Score low if it ignores the JD's key requirements or reads generic."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
        model=_judge_model(),
    )


def cover_letter_quality_metric() -> GEval:
    return GEval(
        name="Cover Letter Quality",
        criteria=(
            "INPUT is a job description. ACTUAL_OUTPUT is a cover letter. Judge: "
            "(1) it opens with a specific hook about the company/role rather than a "
            "generic greeting; (2) it connects the candidate's real experience to "
            "the role's responsibilities; (3) it AVOIDS clichés such as 'I am "
            "writing to express', 'passion for', and 'team player'; (4) it is "
            "concise (a few short paragraphs). Score low if it is generic or uses "
            "the banned clichés."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
        model=_judge_model(),
    )


def interview_prep_no_fabrication_metric() -> GEval:
    """
    Grounding judge for the interview prep guide.

    The prep node is instructed to base STAR / behavioural answers ONLY on the
    candidate's real experience. This judge flags any behavioural story or
    candidate-attributed claim that invents experience absent from the resume.
    Generic technical-knowledge questions and model answers are NOT fabrication;
    only false claims about the candidate's own history are.
    """
    return GEval(
        name="Interview Prep No Fabrication",
        criteria=(
            "CONTEXT is the candidate's original resume. ACTUAL_OUTPUT is an "
            "agent-generated interview prep guide. Judge ONLY the claims the guide "
            "makes about the candidate's own background — especially STAR / "
            "behavioural answers framed in the first person. Score low if any such "
            "answer invents an employer, role, project, team size, metric, degree, "
            "or skill that is NOT in the original resume. Generic technical "
            "questions and model answers describing general knowledge are fine and "
            "should NOT be penalised. Score high only when every first-person claim "
            "about the candidate's experience traces back to the resume."
        ),
        evaluation_params=[
            LLMTestCaseParams.CONTEXT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.8,
        model=_judge_model(),
    )


def faithfulness_metric() -> FaithfulnessMetric:
    # Treats the original resume (retrieval_context) as ground truth and flags
    # claims in the tailored resume that contradict or aren't supported by it.
    return FaithfulnessMetric(threshold=0.8, model=_judge_model())


# ── Deterministic guard (no LLM) ────────────────────────────────────────────────

_PII_TOKEN = re.compile(r"\[PII:[A-Z_]+(?::[0-9a-f]+)?\]")


def check_no_pii_tokens(*texts: str) -> list[str]:
    """Return any stray [PII:*] tokens left in the final documents (should be none)."""
    leaks: list[str] = []
    for t in texts:
        if t:
            leaks.extend(_PII_TOKEN.findall(t))
    return leaks


# ── Test-case builders ──────────────────────────────────────────────────────────

def resume_test_case(job_description: str, original_resume: str, tailored_resume: str) -> LLMTestCase:
    """Test case for the resume-focused metrics."""
    return LLMTestCase(
        input=job_description,
        actual_output=tailored_resume,
        context=[original_resume],
        retrieval_context=[original_resume],
    )


def cover_letter_test_case(job_description: str, cover_letter: str) -> LLMTestCase:
    """Test case for the cover-letter quality metric (uses INPUT + ACTUAL_OUTPUT)."""
    return LLMTestCase(
        input=job_description,
        actual_output=cover_letter,
    )


def grounding_test_case(original_resume: str, document: str, job_description: str = "") -> LLMTestCase:
    """
    Test case for the resume-grounded no-fabrication metrics on any document
    (cover letter, interview prep guide). CONTEXT carries the original resume.
    """
    return LLMTestCase(
        input=job_description or "(grounding check)",
        actual_output=document,
        context=[original_resume],
    )
