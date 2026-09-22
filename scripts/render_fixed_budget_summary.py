#!/usr/bin/env python3
"""Render the complete B1 oracle-only aggregation as reviewable Markdown."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results/supplementary/fixed_budget_9seq_summary.json"
OUTPUT = REPO / "results/supplementary/fixed_budget_9seq_summary.md"


def pct(value: float) -> str:
    return f"{100 * value:.3f}"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    lines = [
        "# B1 Fixed-Total-Candidate-Budget Summary",
        "",
        "This is **candidate-set oracle Recall@1**, not end-to-end retrieval Recall@1. "
        "For each total budget, the split-union condition divides that budget between "
        "the NSD and height indices; the single-index conditions use the same total budget.",
        "",
        "No persisted training-split calibrated score tuple exists for `raw_height240`; "
        "therefore no fixed-score reranking row is reported.",
        "",
        "## Query-weighted aggregate",
        "",
        "| Total budget | NSD index | Height index | Split union | Union gain vs. best single | Queries |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, row in source["results"].items():
        scores = row["query_weighted_candidate_oracle_recall_at_1"]
        lines.append(
            f"| {budget} | {pct(scores['nsd_index'])} | {pct(scores['height_index'])} | "
            f"{pct(scores['split_union'])} | {row['split_union_gain_vs_best_single_pp']:+.3f} pp | "
            f"{row['n_queries']:,} |"
        )

    for budget, row in source["results"].items():
        lines.extend([
            "",
            f"## Per-sequence oracle Recall@1 (total budget {budget})",
            "",
            "| Sequence | Queries | NSD index | Height index | Split union |",
            "|---|---:|---:|---:|---:|",
        ])
        for tag, values in row["per_sequence"].items():
            scores = values["oracle_recall_at_1"]
            lines.append(
                f"| {tag} | {values['n_queries']:,} | {pct(scores['nsd_index'])} | "
                f"{pct(scores['height_index'])} | {pct(scores['split_union'])} |"
            )

    lines.extend([
        "",
        "## Provenance",
        "",
        f"- Source JSON: `{source['provenance']['source_artifact']}`",
        f"- Source SHA-256: `{source['provenance']['source_artifact_sha256']}`",
        f"- Checkpoint: `{source['provenance']['checkpoint']}`",
        f"- Seed: {source['provenance']['seed']}",
        f"- Command: `{source['provenance']['execution_command']}`",
        "",
    ])
    OUTPUT.write_text("\n".join(lines))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
