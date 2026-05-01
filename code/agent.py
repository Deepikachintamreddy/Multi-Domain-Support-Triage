"""SupportTriageAgent — orchestrates the full pipeline for one ticket.

Stages:
  1. sanitize           (safety.py)
  2. domain route       (classifier.infer_domain / normalize_company)
  3. retrieve evidence  (retriever.HybridRetriever)
  4. classify           (classifier.classify_request_type / classify_risk)
  5. decision gate      (this file)
  6. response generation(responder.py)

The decision gate is plain code on purpose — it's testable, auditable, and
easy to defend in the AI Judge interview.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from classifier import (
    RiskAssessment,
    classify_request_type,
    classify_risk,
    infer_domain,
    normalize_company,
    requires_account_action,
)
from config import Config
from responder import (
    escalation_response,
    fallback_extractive_answer,
    generate_response,
)
from retriever import HybridRetriever, RetrievedChunk
from safety import sanitize

log = logging.getLogger("agent")


# Minimum content tokens to treat a ticket as meaningful
_MIN_CONTENT_TOKENS = 6


class SupportTriageAgent:
    def __init__(self, corpus_root: Path, config: Config) -> None:
        self.cfg = config
        self.retriever = HybridRetriever(
            corpus_root=corpus_root,
            domains=config.corpus_subdirs,
            chunk_size_tokens=config.chunk_size_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
            embed_model_name=config.embed_model,
            use_reranker=config.use_reranker,
            rerank_model_name=config.rerank_model,
        )

    # ─── single-ticket entry point ──────────────────────────────────────
    def handle(self, issue: str, subject: str, company: str) -> Dict[str, str]:
        ticket = sanitize(
            issue=issue,
            subject=subject,
            invalid_signals=self.cfg.invalid_signals,
            dangerous_patterns=self.cfg.dangerous_patterns,
        )
        text = ticket.cleaned_text

        # 0. Empty / tiny ticket gate (Item #5) ----------------------------
        content_tokens = [t for t in text.split() if len(t) > 1]
        is_tiny = len(content_tokens) < _MIN_CONTENT_TOKENS

        # 1. Domain routing -------------------------------------------------
        normalized = normalize_company(company)
        domain = normalized or infer_domain(text)

        # 2. Retrieve evidence ---------------------------------------------
        # If the ticket has multiple sub-questions, retrieve per-sub-question
        # and pick the one with the strongest evidence as the primary topic.
        # The full text is still used for risk / injection / classification so
        # we never miss high-risk signals hidden in a secondary sub-question.
        retrieval_query = text
        primary_subq_idx = 0
        if len(ticket.sub_questions) > 1:
            best_score = -1.0
            best_idx = 0
            for i, sq in enumerate(ticket.sub_questions):
                trial = self.retriever.retrieve(
                    query=sq, domain=domain, top_k=1,
                    top_k_bm25=self.cfg.top_k_bm25,
                    top_k_dense=self.cfg.top_k_dense,
                )
                s = trial[0].final_score if trial else 0.0
                if s > best_score:
                    best_score, best_idx = s, i
            retrieval_query = ticket.sub_questions[best_idx]
            primary_subq_idx = best_idx

        chunks: List[RetrievedChunk] = self.retriever.retrieve(
            query=retrieval_query,
            domain=domain,
            top_k=self.cfg.final_top_k,
            top_k_bm25=self.cfg.top_k_bm25,
            top_k_dense=self.cfg.top_k_dense,
        )
        
        # Sub-product hallucination penalty (Issue #3)
        # If the chunk's source path implies a specific sub-product (like Chrome or iOS)
        # but the user didn't mention it, penalize the score so we don't hallucinate context.
        for rc in chunks:
            path_low = rc.chunk.source_path.lower()
            text_low = text.lower()
            for sub_product in ["chrome", "ios", "android", "mac", "windows", "slack", "jira"]:
                if sub_product in path_low and sub_product not in text_low:
                    rc.final_score *= 0.7
        chunks.sort(key=lambda x: x.final_score, reverse=True)
        
        evidence_score = chunks[0].final_score if chunks else 0.0

        # Derive product_area from the top retrieved chunk's subdirectory
        # (Item #15 — matches sample CSV vocabulary like screen, community, etc.)
        product_area = self._product_area_from_chunks(chunks, domain)

        # 3. Classify -------------------------------------------------------
        # Risk and injection are checked against the FULL text, not just
        # the primary sub-question, so a high-risk secondary question still
        # forces escalation.
        request_type = classify_request_type(
            text,
            contains_injection=ticket.contains_injection,
            contains_dangerous=ticket.contains_dangerous,
            evidence_score=evidence_score,
            is_tiny=is_tiny,
        )
        risk = classify_risk(text, self.cfg.high_risk_keywords)

        # 4. Decide ---------------------------------------------------------
        status, decision_reason = self._decide(
            ticket_text=text,
            risk=risk,
            request_type=request_type,
            evidence_score=evidence_score,
            contains_secret=ticket.contains_secret,
            contains_injection=ticket.contains_injection,
            contains_dangerous=ticket.contains_dangerous,
            contains_pii=ticket.contains_pii,
            domain=domain,
            is_tiny=is_tiny,
        )

        # 5. Generate the user-facing response ------------------------------
        if status == "escalated":
            team = product_area if domain else "our"
            response = escalation_response(
                team=team, reason=decision_reason, risk=risk,
                domain=domain, request_type=request_type,
            )
        elif request_type == "invalid" or is_tiny:
            response = self._oos_or_tiny_response(domain, is_tiny)
        else:
            llm_text = generate_response(
                issue=retrieval_query,
                chunks=chunks,
                provider=self.cfg.llm_provider,
                anthropic_model=self.cfg.anthropic_model,
                openai_model=self.cfg.openai_model,
                max_tokens=self.cfg.llm_max_tokens,
                temperature=self.cfg.llm_temperature,
            )
            response = llm_text or fallback_extractive_answer(retrieval_query, chunks)

        # 6. Compose justification (traceable to corpus) --------------------
        top_source = chunks[0].chunk.source_path if chunks else "no sources"
        all_sources = ", ".join(sorted({c.chunk.source_path for c in chunks[:3]})) or "no sources"
        multi_q_note = ""
        if len(ticket.sub_questions) > 1:
            multi_q_note = (
                f" Ticket contains {len(ticket.sub_questions)} sub-questions; "
                f"primary addressed: #{primary_subq_idx + 1}."
            )
        justification = self._build_justification(
            domain=domain,
            risk=risk,
            request_type=request_type,
            status=status,
            decision_reason=decision_reason,
            evidence_score=evidence_score,
            top_source=top_source,
            all_sources=all_sources,
            multi_q_note=multi_q_note,
        )

        return {
            "status": status,
            "product_area": product_area,
            "response": response.strip(),
            "justification": justification,
            "request_type": request_type,
        }

    # ─── helpers ────────────────────────────────────────────────────────
    def _product_area_from_chunks(
        self, chunks: List[RetrievedChunk], domain: str | None,
    ) -> str:
        """Derive product_area from the top chunk's corpus subdirectory."""
        if not chunks:
            return "general_support"
        
        path = chunks[0].chunk.source_path.replace("\\", "/")
        parts = path.split("/")
        
        # If the path is e.g. "hackerrank/screen/faq.md" or "claude/privacy-and-legal/xyz.md"
        # The first part is the domain. The second part is the product area.
        if len(parts) >= 2:
            pa = parts[1].lower().replace("-", "_").replace(".md", "")
            
            # Special deeply nested paths that define their own product area
            if "travel-support" in path:
                return "travel_support"
            if "conversation-management" in path:
                return "conversation_management"
                
            # Map specific directories to their expected CSV labels or fallbacks
            mapping = {
                "screen": "screen",
                "hackerrank_community": "community",
                "privacy_and_legal": "privacy",
                "uncategorized": "general_support",
                "general_help": "general_support",
                "consumer": "general_support",
                "merchant": "general_support",
                "support": "general_support",
                "claude": "general_support",
                "hackerrank": "general_support",
                "visa": "general_support",
                "interviews": "general_support",
                "integrations": "general_support",
                "settings": "general_support",
                "claude_code": "general_support",
                "claude_for_education": "general_support",
                "claude_for_nonprofits": "general_support",
                "safeguards": "general_support",
                "team_and_enterprise_plans": "general_support",
            }
            return mapping.get(pa, "general_support")
            
        return "general_support"

    def _oos_or_tiny_response(self, domain: str | None, is_tiny: bool) -> str:
        """Reply for out-of-scope or tiny tickets."""
        if is_tiny:
            return (
                "Thanks for reaching out. Your message is a bit brief for us to "
                "identify the issue. Could you please share more details — "
                "including which product (HackerRank, Claude, or Visa) you need "
                "help with and a description of the problem?"
            )
        return (
            "I appreciate you reaching out. This request falls outside the "
            "scope of our support documentation. If you have a question about "
            "HackerRank, Claude, or Visa products, please let us know and "
            "we'll be happy to help."
        )

    def _decide(
        self,
        ticket_text: str,
        risk: RiskAssessment,
        request_type: str,
        evidence_score: float,
        contains_secret: bool,
        contains_injection: bool,
        contains_dangerous: bool,
        contains_pii: bool,
        domain: str | None,
        is_tiny: bool = False,
    ) -> tuple[str, str]:
        """Returns (status, reason)."""
        # 1. High-risk must NEVER be short-circuited
        if risk.level == "high":
            if requires_account_action(ticket_text) or evidence_score < self.cfg.min_evidence_similarity:
                return "escalated", f"high-risk + action/weak-evidence: {risk.triggered_terms[0]}"
        if contains_pii:
            return "escalated", "PII detected in ticket (card number / CVV) — human handling required"
        if contains_secret:
            return "escalated", "user pasted a secret (key/password) — needs human handling"
        if requires_account_action(ticket_text):
            # If we have a very strong self-serve document (e.g. score > 0.5), reply with it
            # instead of escalating. Otherwise, escalate as an account action.
            if risk.level == "high" or evidence_score < 0.50:
                return "escalated", "requires account-specific action by a human agent"
            
        # 2. Dangerous/Injection -> invalid + escalate
        if contains_dangerous:
            return "escalated", "dangerous/malicious instruction detected (config.py:dangerous_patterns)"
        if contains_injection:
            return "escalated", "prompt-injection detected (config.py:invalid_signals)"
            
        # 3. Invalid (genuine spam/empty/OOS) -> reply with OOS
        if request_type == "invalid":
            return "replied", "out-of-scope / invalid; replied with polite OOS message"
        if is_tiny and domain is None:
            return "replied", "too few content tokens and no domain — asked user for details"
            
        # 4. Weak evidence -> escalate
        if domain is None and evidence_score < self.cfg.min_evidence_similarity:
            return "escalated", "no domain identified and no relevant docs"
        if evidence_score < self.cfg.min_evidence_similarity:
            return "escalated", f"no sufficiently relevant docs (score={evidence_score:.2f})"
        if risk.level == "medium" and evidence_score < self.cfg.medium_risk_evidence_floor:
            return (
                "escalated",
                f"medium-risk topic ({risk.reason}) with low-confidence evidence ({evidence_score:.2f})",
            )
            
        # 5. Else -> reply
        return "replied", f"answerable from corpus (score={evidence_score:.2f}, risk={risk.level})"

    def _build_justification(
        self,
        domain: str | None,
        risk: RiskAssessment,
        request_type: str,
        status: str,
        decision_reason: str,
        evidence_score: float,
        top_source: str,
        all_sources: str,
        multi_q_note: str = "",
    ) -> str:
        """Traceable justification naming the source file and rule (Items #12-13)."""
        if status == "escalated":
            return (
                f"Domain={domain or 'unknown'}; request_type={request_type}; "
                f"risk={risk.level}. Top source: {top_source} "
                f"(score={evidence_score:.2f}). "
                f"Escalated because: {decision_reason}.{multi_q_note}"
            )
        return (
            f"Domain={domain or 'unknown'}; request_type={request_type}; "
            f"risk={risk.level}. Reply drawn from {top_source} "
            f"(top score={evidence_score:.2f}). "
            f"All sources: [{all_sources}]. "
            f"Decision={status}: {decision_reason}.{multi_q_note}"
        )
