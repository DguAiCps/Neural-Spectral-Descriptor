#!/usr/bin/env python3
"""Check that the paper-table provenance manifest matches this checkout."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs" / "reproducibility_manifest.yaml"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    data = yaml.safe_load(MANIFEST.read_text())
    tex = REPO / data["paper_source"]
    tables = data["tables"]
    # In the code-only release bundle the paper source and .git metadata are
    # absent; those cross-checks are skipped there and enforced in the dev tree.
    bundle_mode = not tex.exists()
    if bundle_mode:
        print("note: paper source not present (code-only bundle); skipping table-label cross-check")
    else:
        labels = set(re.findall(r"\\label\{(tab:[^}]+)\}", tex.read_text()))
        if labels != set(tables):
            raise SystemExit(
                f"table-label mismatch; missing={sorted(labels - set(tables))}, "
                f"extra={sorted(set(tables) - labels)}"
            )

    required = {"scope", "config", "command", "checkpoints", "result_artifacts",
                "seed_policy", "external_commit"}
    incomplete = [name for name, entry in tables.items() if required - set(entry)]
    if incomplete:
        raise SystemExit(f"incomplete table entries: {', '.join(incomplete)}")

    for name, source in data["external_repositories"].items():
        path = REPO / source["path"]
        if bundle_mode:
            continue
        if not path.is_dir():
            raise SystemExit(f"missing external repository {name}: {path}")
        actual = git_head(path)
        if actual != source["commit"]:
            raise SystemExit(
                f"external commit mismatch for {name}: expected {source['commit']}, got {actual}"
            )
    if bundle_mode:
        print("note: external-repository commit checks skipped (code-only bundle)")
    print(f"reproducibility manifest passed ({len(tables)} tables)")


if __name__ == "__main__":
    main()
