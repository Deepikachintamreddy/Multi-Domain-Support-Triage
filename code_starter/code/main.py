"""
Multi-Domain Support Triage — entry point.

Run:
    python -m code.main \
        --input  ../support_issues/support_issues.csv \
        --output ../support_issues/output.csv \
        --corpus ../data
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from agent import SupportTriageAgent
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("main")


# Output columns required by the problem statement, in this order.
OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    p.add_argument("--input", required=True, help="Path to support_tickets.csv")
    p.add_argument("--output", required=True, help="Path to write output.csv")
    p.add_argument("--corpus", required=True, help="Path to data/ corpus root")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (useful while tuning).",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="Run against sample_support_issues.csv and print per-row diff.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config()

    input_path = Path(args.input)
    output_path = Path(args.output)
    corpus_path = Path(args.corpus)

    if not input_path.exists():
        log.error("Input CSV not found: %s", input_path)
        return 1
    if not corpus_path.exists():
        log.error("Corpus directory not found: %s", corpus_path)
        return 1

    log.info("Loading input CSV: %s", input_path)
    df = pd.read_csv(input_path)
    log.info("Loaded %d rows. Columns: %s", len(df), list(df.columns))

    # Remember original column names so we can preserve them in output.
    original_columns = list(df.columns)

    # Normalize column names to lowercase to handle CSV variations
    df.columns = [col.lower() for col in df.columns]

    # Strict validation against problem_statement.md input schema.
    required_cols = {"issue", "subject", "company"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(
            "Input CSV is missing required columns: %s. "
            "Expected at minimum: issue, subject, company.",
            sorted(missing),
        )
        return 2

    if args.limit:
        df = df.head(args.limit).copy()
        log.info("Limited to first %d rows.", args.limit)

    log.info("Initialising agent (this builds the retriever index)...")
    agent = SupportTriageAgent(corpus_root=corpus_path, config=cfg)

    log.info("Triaging %d tickets...", len(df))
    out_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="triage"):
        result = agent.handle(
            issue=str(row.get("issue", "") or ""),
            subject=str(row.get("subject", "") or ""),
            company=str(row.get("company", "") or "None"),
        )
        out_rows.append(result)

    out_df = pd.DataFrame(out_rows, columns=OUTPUT_COLUMNS)
    # Restore original column names for the input columns.
    input_df = df.copy()
    input_df.columns = original_columns
    final_df = pd.concat([input_df.reset_index(drop=True), out_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    log.info("Wrote %d rows to %s", len(final_df), output_path)

    # Sanity validations
    bad_status = ~final_df["status"].isin(["replied", "escalated"])
    bad_rt = ~final_df["request_type"].isin(
        ["product_issue", "feature_request", "bug", "invalid"]
    )
    if bad_status.any():
        log.error("Invalid status values in %d rows! Aborting.", int(bad_status.sum()))
        return 3
    if bad_rt.any():
        log.error("Invalid request_type values in %d rows! Aborting.", int(bad_rt.sum()))
        return 3
    if final_df["response"].isna().any() or (final_df["response"] == "").any():
        log.warning("Some response cells are empty.")

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
