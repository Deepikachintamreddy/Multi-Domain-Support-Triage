"""Pre-processing: prompt injection detection, PII scrubbing, sub-question split."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


# ── PII patterns ────────────────────────────────────────────────────────
# Conservative — these will produce false positives on normal text;
# we only use them as flags, not to block.
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_CVV_NEAR_CARD_RE = re.compile(r"\bcvv\b\s*[:#]?\s*\d{3,4}", re.IGNORECASE)
_PASSWORD_RE = re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,})\b")


@dataclass
class SanitizedTicket:
    raw_text: str
    cleaned_text: str
    contains_pii: bool
    contains_secret: bool
    contains_injection: bool
    sub_questions: List[str]


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def detect_injection(text: str, signals: set[str]) -> bool:
    low = text.lower()
    return any(sig in low for sig in signals)


def detect_pii(text: str) -> bool:
    if _CARD_RE.search(text):
        return True
    if _CVV_NEAR_CARD_RE.search(text):
        return True
    if _PASSWORD_RE.search(text):
        return True
    return False


def detect_secret(text: str) -> bool:
    return bool(_API_KEY_RE.search(text))


def split_sub_questions(text: str) -> List[str]:
    """Very light splitter: separates by question marks and explicit numbering."""
    if not text:
        return []
    # If the ticket is clearly numbered ("1.", "2.", "3."), split that way
    numbered = re.split(r"(?:^|\n)\s*\d+[.\)]\s+", text)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) > 1:
        return numbered
    # Otherwise split by '?'
    parts = [p.strip() for p in re.split(r"\?+", text) if p.strip()]
    if len(parts) > 1:
        # re-add the '?' for nicer prompting
        return [p + "?" if not p.endswith("?") else p for p in parts]
    return [text.strip()]


def sanitize(
    issue: str,
    subject: str,
    invalid_signals: set[str],
) -> SanitizedTicket:
    raw = " ".join(filter(None, [subject, issue]))
    cleaned = _normalise_ws(_strip_html(raw))
    return SanitizedTicket(
        raw_text=raw,
        cleaned_text=cleaned,
        contains_pii=detect_pii(cleaned),
        contains_secret=detect_secret(cleaned),
        contains_injection=detect_injection(cleaned, invalid_signals),
        sub_questions=split_sub_questions(cleaned),
    )
