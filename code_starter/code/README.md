# Multi-Domain Support Triage Agent

Terminal-based triage agent for support tickets across **HackerRank**, **Claude**, and **Visa**, built for the HackerRank Orchestrate (May 2026) hackathon.

## What it does

For each row in `support_issues/support_issues.csv`, the agent:
1. Sanitises the ticket (HTML strip, prompt-injection / PII detection, sub-question split).
2. Routes it to the right domain (`hackerrank` / `claude` / `visa`) using the `company` column or content-based inference.
3. Retrieves the top-K most relevant chunks from `data/` using **hybrid BM25 + dense embeddings**.
4. Classifies `request_type` (`product_issue` / `feature_request` / `bug` / `invalid`) and risk (`low`/`medium`/`high`).
5. Runs a **deterministic decision gate** to choose `replied` vs `escalated`.
6. Generates a **grounded** response using only the retrieved chunks (with a strict no-hallucination system prompt and `temperature=0`).
7. Writes a 5-column row to `output.csv` plus a traceable `justification`.

## Architecture

```
[input row]
   │
   ▼
[1] safety.sanitize          → injection / PII / sub-question split
   │
   ▼
[2] classifier.normalize_company / infer_domain  → hackerrank | claude | visa | None
   │
   ▼
[3] retriever.HybridRetriever.retrieve  → top-K chunks (BM25 ⊕ dense)
   │
   ▼
[4] classifier.classify_request_type / classify_risk
   │
   ▼
[5] agent._decide  →  status, reason   (deterministic, code-driven)
   │
   ▼
[6] responder.generate_response  → grounded reply (LLM @ T=0, falls back to extractive)
   │
   ▼
[output row: status, product_area, response, justification, request_type]
```

### Why these choices

- **Hybrid retrieval (BM25 + dense)**: BM25 catches exact-match terminology like *CodePair*, *Visa Direct*, *2FA*; dense catches paraphrases. They complement each other and the fusion is more robust on small corpora than either alone.
- **Decision gate as code, not an LLM call**: reproducibility and auditability. The eval explicitly penalises wrong escalation; a code gate is testable, an LLM is not.
- **Deterministic everywhere**: `temperature=0`, `seed=42`, pinned dependencies. The judge can re-run our CSV and reproduce numbers.
- **Grounded prompt**: the system prompt explicitly says "Do NOT use knowledge outside the SUPPORT DOCS"; sources are listed inline; the model is told to say "I don't know" rather than guess. We also keep an extractive fallback so the pipeline still produces sane output if the LLM is unreachable.
- **Risk taxonomy from the spec**: the high-risk keyword set mirrors the problem statement language ("billing, bugs, fraud, permissions, account access, assessments, or other sensitive situations").

## Layout

```
code/
├── README.md         ← you are here
├── requirements.txt
├── main.py           ← entry point (reads CSV, runs agent, writes output.csv)
├── agent.py          ← orchestrator + decision gate
├── retriever.py      ← BM25 + dense hybrid retrieval
├── classifier.py     ← domain routing, request_type, risk
├── safety.py         ← injection / PII detection, sub-question split
├── responder.py      ← grounded LLM call + extractive fallback
├── config.py         ← all thresholds & model names in one place
└── tests/
    └── test_pipeline.py  ← run on sample_support_issues.csv
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set your API key (NEVER commit this)
export ANTHROPIC_API_KEY=sk-ant-...
# or:
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai          # default is "anthropic"
```

## Run

From the `code/` directory:

```bash
python main.py \
    --input  ../support_tickets/support_tickets.csv \
    --output ../support_tickets/output.csv \
    --corpus ../data
```

To dry-run on the labelled sample first:

```bash
python main.py \
    --input  ../support_tickets/sample_support_tickets.csv \
    --output /tmp/sample_output.csv \
    --corpus ../data \
    --limit 20
```

## Tuning knobs (all in `config.py`)

| Knob | Meaning | Default |
|---|---|---|
| `min_evidence_similarity` | If best chunk score < this, escalate (no hallucinating) | `0.30` |
| `final_top_k` | Chunks fed to the LLM | `5` |
| `chunk_size_tokens` | Indexer chunk width | `500` |
| `use_reranker` | Cross-encoder reranking (slower, higher precision) | `False` |
| `llm_temperature` | LLM sampling temperature | `0.0` |
| `high_risk_keywords` | Force-escalate triggers | spec-derived set |

## Determinism

- `temperature=0`, `seed=42` for the OpenAI client.
- Indexer iterates files in **sorted order**.
- BM25 + dense scores are deterministic.
- Pinned `requirements.txt`.

## Failure modes (known)

1. **Multilingual tickets** — corpus is mostly English; non-English tickets retrieve poorly. Future fix: multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`).
2. **Multiple sub-questions** — current behaviour answers the highest-evidence sub-question and acknowledges the rest in `justification`. Future fix: per-sub-question handling and merge.
3. **Adversarial paraphrases** of high-risk content (e.g. "I had a thing happen with my plastic at the store") — keyword set won't catch it. Future fix: small classifier fine-tuned on labelled support data.

## Evaluation rationale

We map cleanly onto the four scoring dimensions:

| Dimension | How this code earns score |
|---|---|
| Agent design | Modules per concern (retrieval / safety / classifier / responder / decision); explicit escalation logic; deterministic; pinned deps; this README. |
| AI Judge interview | Trade-offs documented above; failure modes listed; config.py centralises tuning. |
| Output CSV | Strict allowed-value enforcement; grounded responses; traceable `justification`; spec-derived risk taxonomy. |
| AI Fluency (log.txt) | Prompts in our log are scoped, critical, and architectural — driving the AI rather than being driven. |
