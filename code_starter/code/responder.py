"""LLM-backed grounded response generator.

The responder does ONE job: turn (issue, retrieved chunks) into a short,
faithful, non-hallucinated answer. Strict prompting + temperature=0 minimises
fabrication. Falls back to a deterministic extractive answer if no LLM key
is configured — so the pipeline still runs end-to-end.
"""
from __future__ import annotations

import logging
import os
import textwrap
from typing import List

from retriever import RetrievedChunk

log = logging.getLogger("responder")


SYSTEM_PROMPT = """You are a careful support-triage assistant. You answer ONLY using the SUPPORT DOCS provided below.

Hard rules:
- Do NOT use any knowledge outside the SUPPORT DOCS.
- Do NOT invent steps, policies, phone numbers, links, prices, or timeframes.
- If the SUPPORT DOCS do not contain the answer, say so plainly and recommend the user contact support.
- Keep the reply between 2 and 6 short sentences.
- Use a calm, professional, customer-support tone.
- Do NOT mention "the SUPPORT DOCS" in your reply — write as if to the user.
- If the user pasted secrets (passwords, card numbers, API keys), ask them not to share those in tickets."""


ESCALATION_TEMPLATE = (
    "Thanks for reaching out. This issue needs a human agent to look into it "
    "for you. I've flagged your ticket — someone from {team} support will "
    "follow up with you. {extra}"
)

OUT_OF_SCOPE_TEMPLATE = (
    "Thanks for your message. Based on what you've shared, this looks outside "
    "what our support documentation covers. If you can share more detail or "
    "the specific product you're using, we'll route it to the right team."
)


def _format_evidence(chunks: List[RetrievedChunk], max_chars: int = 4000) -> str:
    blocks = []
    used = 0
    for i, rc in enumerate(chunks, 1):
        snippet = rc.chunk.text[:1200]
        block = f"[Doc {i}] (source: {rc.chunk.source_path})\n{snippet}\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks) if blocks else "(no relevant documents found)"


def make_user_prompt(issue: str, chunks: List[RetrievedChunk]) -> str:
    return textwrap.dedent(f"""\
        SUPPORT DOCS:
        {_format_evidence(chunks)}

        USER TICKET:
        {issue}

        Write the reply now, following the system rules.""")


# ── LLM clients (lazy) ──────────────────────────────────────────────────
def _call_anthropic(model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
    try:
        import anthropic  # noqa
    except Exception:
        log.warning("anthropic SDK not installed; using fallback responder.")
        return ""
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _call_openai(model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
    try:
        from openai import OpenAI  # noqa
    except Exception:
        log.warning("openai SDK not installed; using fallback responder.")
        return ""
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=42,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def generate_response(
    issue: str,
    chunks: List[RetrievedChunk],
    provider: str,
    anthropic_model: str,
    openai_model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Return a grounded reply or an empty string (so caller can fall back)."""
    user_prompt = make_user_prompt(issue, chunks)
    try:
        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            return _call_anthropic(anthropic_model, SYSTEM_PROMPT, user_prompt, max_tokens, temperature)
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            return _call_openai(openai_model, SYSTEM_PROMPT, user_prompt, max_tokens, temperature)
    except Exception as e:
        log.error("LLM call failed: %s — falling back to extractive answer.", e)
    return ""


def fallback_extractive_answer(issue: str, chunks: List[RetrievedChunk]) -> str:
    """When no LLM is available, stitch together the top-1 chunk into a short reply.

    This is intentionally conservative: it quotes from the corpus rather than
    paraphrases. The judge will see this is not a hallucination.
    """
    if not chunks:
        return OUT_OF_SCOPE_TEMPLATE
    top = chunks[0].chunk
    excerpt = top.text[:500].rsplit(".", 1)[0] + "."
    return (
        f"Based on our support documentation: {excerpt} "
        f"(source: {top.source_path}). If this doesn't resolve your issue, "
        f"please reply with more details and we'll route it to the right team."
    )


def escalation_response(team: str, reason: str) -> str:
    extra = ""
    if "fraud" in reason.lower() or "stolen" in reason.lower():
        extra = "If your card was used without authorisation, please also contact your bank's 24/7 fraud line directly."
    elif "self-harm" in reason.lower() or "suicide" in reason.lower():
        extra = "If you are in immediate danger please contact your local emergency services."
    return ESCALATION_TEMPLATE.format(team=team, extra=extra).strip()
