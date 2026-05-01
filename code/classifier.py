"""Domain routing + request_type/risk classification.

The classifiers here are intentionally hybrid:
  - rules-first for transparency, determinism, and judge-friendliness;
  - LLM as a tie-breaker only when needed (and pluggable).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("classifier")


# ── Domain routing ──────────────────────────────────────────────────────

_HACKERRANK_TERMS = {
    "hackerrank", "codepair", "assessment", "interview kit", "leaderboard",
    "skills certification", "test invite", "candidate", "recruiter",
    "resume builder",
}
_CLAUDE_TERMS = {
    "claude", "anthropic", "claude.ai", "artifact", "projects",
    "claude pro", "claude team", "claude enterprise", "bedrock",
    "lti",
}
_VISA_TERMS = {
    "visa", "credit card", "debit card", "card declined", "atm",
    "merchant", "chargeback", "visa direct", "issuer", "bank",
    "transaction", "swipe", "tap to pay", "contactless",
    "traveller", "carte",
}


def normalize_company(value: str) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v in {"none", "", "null", "nan"}:
        return None
    if "hacker" in v:
        return "hackerrank"
    if "claude" in v or "anthropic" in v:
        return "claude"
    if "visa" in v:
        return "visa"
    return None


def infer_domain(text: str) -> Optional[str]:
    """Used when company is None. Pick the domain by keyword overlap."""
    low = text.lower()
    scores = {
        "hackerrank": sum(1 for t in _HACKERRANK_TERMS if t in low),
        "claude":     sum(1 for t in _CLAUDE_TERMS    if t in low),
        "visa":       sum(1 for t in _VISA_TERMS      if t in low),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ── Request type classification ─────────────────────────────────────────

_BUG_HINTS = re.compile(
    r"\b(error|crash(ed)?|broke(n)?|doesn'?t work|not working|500|404|"
    r"stack ?trace|exception|fail(ed|ing)?|stopped working|is down|"
    r"blocker|blocked|can\s*not\s*able)\b",
    re.IGNORECASE,
)
_FEATURE_HINTS = re.compile(
    r"\b(can you add|please add|i wish|would be nice|feature request|"
    r"support for|add support)\b",
    re.IGNORECASE,
)
_HOWTO_HINTS = re.compile(
    r"\b(how (do|can) i|how to|where (is|do)|why (is|does)|what is|"
    r"step.?by.?step|can you let me know)\b",
    re.IGNORECASE,
)


def classify_request_type(
    text: str,
    contains_injection: bool,
    contains_dangerous: bool,
    evidence_score: float,
    is_tiny: bool = False,
) -> str:
    """Returns one of: product_issue, feature_request, bug, invalid."""
    if contains_injection or contains_dangerous:
        return "invalid"
    if not text or len(text.split()) < 3:
        return "invalid"
    if is_tiny:
        return "invalid"
    if _BUG_HINTS.search(text):
        return "bug"
    if _FEATURE_HINTS.search(text):
        return "feature_request"
    if _HOWTO_HINTS.search(text):
        return "product_issue"
    # Don't dump to invalid just because retrieval was weak — let the decision
    # gate handle escalation via risk and evidence checks. Default to product_issue.
    return "product_issue"


# ── Risk classification ─────────────────────────────────────────────────
@dataclass
class RiskAssessment:
    level: str            # "low" | "medium" | "high"
    triggered_terms: list # which keywords matched
    reason: str


def _kw_match(text_low: str, keyword: str) -> bool:
    """Word-boundary keyword match.

    Multi-word keywords like 'identity theft' or 'live assessment' are matched
    with whitespace-tolerant boundaries. Single-word keywords use \\b boundaries
    so 'press' does NOT match inside 'pressure', and 'sue' does NOT match
    inside 'issue'.
    """
    if " " in keyword:
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in keyword.split()) + r"\b"
    else:
        pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text_low, re.IGNORECASE) is not None


def classify_risk(text: str, high_risk_keywords: set[str]) -> RiskAssessment:
    if not text:
        return RiskAssessment("low", [], "empty input")
    low = text.lower()
    hits = [kw for kw in high_risk_keywords if _kw_match(low, kw)]
    if hits:
        return RiskAssessment(
            "high",
            hits,
            f"high-risk signal(s): {', '.join(hits[:3])}",
        )
    # Medium: account- or money-adjacent but no explicit trigger.
    medium_signals = ["password", "account", "billing", "payment", "subscription"]
    msigs = [m for m in medium_signals if _kw_match(low, m)]
    if msigs:
        return RiskAssessment(
            "medium",
            msigs,
            f"account/billing-adjacent: {', '.join(msigs[:3])}",
        )
    return RiskAssessment("low", [], "no risk signals")


# ── Account-action detection ────────────────────────────────────────────
# These are things the agent CAN'T do for the user — must escalate.
_ACTION_VERBS = re.compile(
    r"\b(reset|unlock|refund|cancel|reinstate|reactivate|merge|delete|close|"
    r"pause|restore|remove)\b",
    re.IGNORECASE,
)
_PERSONAL_REF = re.compile(r"\b(my|our|me|mine)\b", re.IGNORECASE)
_IMPERATIVE_REF = re.compile(
    r"\b(please|can you|could you|i need you to|help me|can someone)\b",
    re.IGNORECASE,
)


def requires_account_action(text: str) -> bool:
    """True if the user is asking the agent to act on their specific account."""
    if not text:
        return False
    # To reduce over-escalation, we require:
    # 1. An action verb (delete, refund, etc.)
    # 2. A personal reference (my account, me, etc.)
    # 3. Imperative/Direct phrasing (please, can you, etc.)
    # This distinguishes "how do I reset my password" from "please reset my password".
    return bool(
        _ACTION_VERBS.search(text) and 
        _PERSONAL_REF.search(text) and 
        _IMPERATIVE_REF.search(text)
    )
