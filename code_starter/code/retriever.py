"""Hybrid (BM25 + dense embeddings) retriever over the local support corpus.

The corpus is in `data/{hackerrank,claude,visa}/`. We index once at startup
into in-memory structures so retrieval has no external dependency and is fast
enough for the whole run. No live web calls — required by the spec.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

log = logging.getLogger("retriever")

# Lazy-import sentence-transformers so the module loads even if it is missing
# during dry-run / lint passes.
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False


# ── Chunk model ─────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: int
    domain: str           # "hackerrank" | "claude" | "visa"
    source_path: str      # relative to corpus root
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    bm25_score: float
    dense_score: float
    final_score: float


# ── Helpers ─────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return None


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _chunk_text(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    # Approximate token count via whitespace split — good enough for chunking.
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    step = max(1, chunk_size_tokens - overlap_tokens)
    for start in range(0, len(tokens), step):
        end = start + chunk_size_tokens
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
    return chunks


# ── Retriever ───────────────────────────────────────────────────────────
class HybridRetriever:
    """In-memory hybrid BM25 + dense retriever, scoped per-domain."""

    SUPPORTED_EXTS = {".md", ".markdown", ".txt", ".html", ".htm", ".json"}

    def __init__(
        self,
        corpus_root: Path,
        domains: Iterable[str],
        chunk_size_tokens: int = 500,
        overlap_tokens: int = 50,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.corpus_root = corpus_root
        self.domains = list(domains)
        self.chunk_size = chunk_size_tokens
        self.overlap = overlap_tokens
        self.embed_model_name = embed_model_name

        self.chunks: List[Chunk] = []
        self._bm25_indexes: Dict[str, Tuple[BM25Okapi, List[int]]] = {}
        self._dense_vectors: Dict[str, Tuple[np.ndarray, List[int]]] = {}
        self._embed_model = None

        self._build()

    # ─── Index build ────────────────────────────────────────────────────
    def _build(self) -> None:
        log.info("Indexing corpus from %s ...", self.corpus_root)
        next_id = 0
        per_domain_chunks: Dict[str, List[Chunk]] = {d: [] for d in self.domains}

        for domain in self.domains:
            ddir = self.corpus_root / domain
            if not ddir.exists():
                log.warning("Domain dir missing: %s", ddir)
                continue
            for path in sorted(ddir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in self.SUPPORTED_EXTS:
                    continue
                text = _read_text_file(path)
                if not text:
                    continue
                if path.suffix.lower() in {".html", ".htm"}:
                    text = _strip_html(text)
                rel = str(path.relative_to(self.corpus_root))
                for chunk_text in _chunk_text(text, self.chunk_size, self.overlap):
                    if len(chunk_text.split()) < 5:
                        continue
                    c = Chunk(
                        chunk_id=next_id,
                        domain=domain,
                        source_path=rel,
                        text=chunk_text,
                    )
                    self.chunks.append(c)
                    per_domain_chunks[domain].append(c)
                    next_id += 1

        log.info(
            "Indexed %d chunks across %d domains.",
            len(self.chunks), len(per_domain_chunks),
        )

        # BM25 per domain
        for domain, dchunks in per_domain_chunks.items():
            if not dchunks:
                continue
            tokenized = [_tokenize(c.text) for c in dchunks]
            ids = [c.chunk_id for c in dchunks]
            self._bm25_indexes[domain] = (BM25Okapi(tokenized), ids)

        # Dense embeddings per domain
        if _ST_AVAILABLE:
            log.info("Loading embedding model: %s", self.embed_model_name)
            self._embed_model = SentenceTransformer(self.embed_model_name)
            for domain, dchunks in per_domain_chunks.items():
                if not dchunks:
                    continue
                texts = [c.text for c in dchunks]
                ids = [c.chunk_id for c in dchunks]
                vecs = self._embed_model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
                self._dense_vectors[domain] = (np.asarray(vecs, dtype=np.float32), ids)
        else:
            log.warning(
                "sentence-transformers not installed — falling back to BM25 only."
            )

    # ─── Query ──────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
        top_k_bm25: int = 50,
        top_k_dense: int = 50,
    ) -> List[RetrievedChunk]:
        """If domain is None we search all domains and let the scores rank."""
        domains = [domain] if domain else self.domains

        # Collect candidate scores keyed by chunk_id.
        bm25_scores: Dict[int, float] = {}
        dense_scores: Dict[int, float] = {}

        for d in domains:
            if d in self._bm25_indexes:
                bm25, ids = self._bm25_indexes[d]
                scores = bm25.get_scores(_tokenize(query))
                # Top-k for this domain
                idx_sorted = np.argsort(scores)[::-1][:top_k_bm25]
                for i in idx_sorted:
                    bm25_scores[ids[i]] = float(scores[i])
            if d in self._dense_vectors and self._embed_model is not None:
                mat, ids = self._dense_vectors[d]
                qvec = self._embed_model.encode(
                    [query], normalize_embeddings=True, show_progress_bar=False
                )[0]
                sims = mat @ qvec
                idx_sorted = np.argsort(sims)[::-1][:top_k_dense]
                for i in idx_sorted:
                    dense_scores[ids[i]] = float(sims[i])

        # Normalize bm25 scores into [0,1] for fusion
        if bm25_scores:
            mx = max(bm25_scores.values()) or 1.0
            bm25_scores = {k: v / mx for k, v in bm25_scores.items()}

        candidates = set(bm25_scores) | set(dense_scores)
        results: List[RetrievedChunk] = []
        chunk_lookup = {c.chunk_id: c for c in self.chunks}
        for cid in candidates:
            b = bm25_scores.get(cid, 0.0)
            d_ = dense_scores.get(cid, 0.0)
            # 50/50 fusion. Tunable.
            final = 0.5 * b + 0.5 * d_
            results.append(
                RetrievedChunk(
                    chunk=chunk_lookup[cid],
                    bm25_score=b,
                    dense_score=d_,
                    final_score=final,
                )
            )
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    def max_dense_similarity(self, query: str, domain: Optional[str] = None) -> float:
        """Quick heuristic for the 'no relevant evidence' gate."""
        if not self._embed_model or not self._dense_vectors:
            return 0.0
        qvec = self._embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        domains = [domain] if domain else self.domains
        best = 0.0
        for d in domains:
            if d not in self._dense_vectors:
                continue
            mat, _ = self._dense_vectors[d]
            best = max(best, float(np.max(mat @ qvec)))
        return best
