"""
guardrails/pii_guard.py — PII detection, redaction, and restoration.

Flow
────
1. scan_pii()     → find all PII in a text, return {type: [values]}
2. redact_pii()   → replace values with [PII:TYPE:n] tokens, return (clean_text, mapping)
3. restore_pii()  → swap tokens back to real values in final output docs
4. sanitize_log() → one-way redact for log strings (no restore needed)

Sensitive-tier PII (SSN, credit card, passport) raises PIIBlockedError —
the agent halts rather than sending that data to any LLM.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────────

class PIIBlockedError(ValueError):
    """Raised when highly-sensitive PII is detected that must not leave the device."""


# ─────────────────────────────────────────────────────────────────────────────
# Pattern registry
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (pii_type, compiled_regex, tier)
# tier "sensitive" → block outright; tier "standard" → redact + track
_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # ── Sensitive-tier ────────────────────────────────────────────────────────
    ("SSN",         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                          "sensitive"),
    ("CREDIT_CARD", re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"), "sensitive"),
    ("PASSPORT",    re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),                          "sensitive"),

    # ── Standard-tier ─────────────────────────────────────────────────────────
    ("EMAIL",       re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),                "standard"),
    ("PHONE",       re.compile(
        r"(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),            "standard"),
    ("LINKEDIN",    re.compile(
        r"https?://(?:www\.)?linkedin\.com/in/[\w\-]+/?",
        re.IGNORECASE),                                                             "standard"),
    ("GITHUB",      re.compile(
        r"https?://(?:www\.)?github\.com/[\w\-]+/?",
        re.IGNORECASE),                                                             "standard"),
    ("URL",         re.compile(
        r"https?://[^\s,\"'<>]+",
        re.IGNORECASE),                                                             "standard"),
    # Street address heuristic: "123 Main St" / "456 Oak Avenue, City"
    ("ADDRESS",     re.compile(
        r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
        r"\s+(?:St(?:reet)?|Ave(?:nue)?|Blvd|Rd|Road|Dr(?:ive)?|Ln|Lane|"
        r"Ct|Court|Pl(?:ace)?|Way|Pkwy|Parkway|Cir(?:cle)?)\b",
        re.IGNORECASE),                                                             "standard"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PIIReport:
    """Summary of PII found — stores types+counts, never the raw values."""
    counts: dict[str, int] = field(default_factory=dict)
    has_sensitive: bool = False
    sensitive_types: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.counts:
            return "No PII detected."
        parts = [f"{t}: {n}" for t, n in sorted(self.counts.items())]
        return "PII detected — " + ", ".join(parts)


def scan_pii(text: str) -> tuple[PIIReport, dict[str, list[str]]]:
    """
    Scan *text* for PII.

    Returns
    -------
    report   : PIIReport — safe to log (no raw values)
    findings : dict[pii_type, list[matched_strings]] — contains raw values,
               used internally for redaction; never log this directly.
    """
    findings: dict[str, list[str]] = {}
    report = PIIReport()

    for pii_type, pattern, tier in _PATTERNS:
        matches = pattern.findall(text)
        # findall returns strings or tuples depending on groups
        flat = [m if isinstance(m, str) else m[0] for m in matches]
        flat = [m.strip() for m in flat if m.strip()]
        if flat:
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique = [x for x in flat if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
            findings[pii_type] = unique
            report.counts[pii_type] = len(unique)
            if tier == "sensitive":
                report.has_sensitive = True
                report.sensitive_types.append(pii_type)

    return report, findings


def redact_pii(
    text: str,
    findings: Optional[dict[str, list[str]]] = None,
) -> tuple[str, dict[str, str]]:
    """
    Replace PII values in *text* with reversible tokens ``[PII:TYPE:uid]``.

    Parameters
    ----------
    text     : the original text
    findings : pre-computed findings from scan_pii(); if None, scan_pii runs again.

    Returns
    -------
    redacted : text with PII replaced by tokens
    mapping  : {token → original_value} used by restore_pii()
    """
    if findings is None:
        _, findings = scan_pii(text)

    mapping: dict[str, str] = {}
    redacted = text

    for pii_type, values in findings.items():
        for value in values:
            token = f"[PII:{pii_type}:{uuid.uuid4().hex[:8]}]"
            mapping[token] = value
            # Replace all occurrences of this exact value
            redacted = redacted.replace(value, token)

    return redacted, mapping


def restore_pii(text: str, mapping: dict[str, str]) -> str:
    """
    Swap ``[PII:TYPE:uid]`` tokens back to their original values.
    Safe to call on the final tailored resume / cover letter.
    """
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


# Matches any leftover [PII:...] token the LLM may have invented (no uid suffix)
_STRAY_PII_TOKEN = re.compile(r"\[PII:[A-Z_]+(?::[a-f0-9]+)?\]")


def strip_stray_pii_tokens(text: str) -> str:
    """
    Remove any ``[PII:*]`` tokens that were not in the mapping and therefore
    not restored by restore_pii().  These are hallucinated tokens the LLM
    invented; stripping them is safer than leaving the literal placeholder text
    in the final document.
    """
    return _STRAY_PII_TOKEN.sub("", text)


def sanitize_log(text: str) -> str:
    """
    One-way redaction for log messages — replaces PII with ``[REDACTED:TYPE]``.
    Not reversible; use only on strings that are going to be logged / stored
    in state messages, not on documents that will be shown to the user.
    """
    result = text
    for pii_type, pattern, _ in _PATTERNS:
        result = pattern.sub(f"[REDACTED:{pii_type}]", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: check & raise on sensitive PII
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_sensitive_pii(report: PIIReport, source: str = "input") -> None:
    """
    Raise PIIBlockedError if *report* contains sensitive-tier PII.
    Call this before any data leaves the local process.
    """
    if report.has_sensitive:
        types = ", ".join(report.sensitive_types)
        raise PIIBlockedError(
            f"Highly-sensitive PII found in {source}: {types}. "
            "Remove this information before running the agent."
        )
