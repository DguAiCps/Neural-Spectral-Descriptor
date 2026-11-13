#!/usr/bin/env python3
"""저장한 CVPR PDF의 그림 캡션과 원문 페이지 미리보기를 추출한다."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1] / "docs/related works/CVPR_2024-2026_NSD"


def main() -> None:
    """그림 번호는 PDF 텍스트에서 읽고 미리보기는 원문 페이지를 렌더한다."""
    inventory = []
    for path in sorted((ROOT / "papers").glob("*/main.pdf")):
        rows = []
        with pymupdf.open(path) as document:
            for page_index, page in enumerate(document):
                blocks = page.get_text("blocks")
                for block in blocks:
                    if len(block) < 7 or block[6] != 0:
                        continue
                    text = re.sub(r"\s+", " ", block[4]).strip()
                    match = re.search(r"\bFigure\s+(\d+)[.:]\s*(.*)", text)
                    if not match:
                        continue
                    if any(row["figure"] == int(match[1]) for row in rows):
                        continue
                    image = ROOT / "previews" / path.parent.name / f"p{page_index + 1:02d}.png"
                    if not image.exists():
                        image.parent.mkdir(parents=True, exist_ok=True)
                        page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False, annots=False).save(image)
                    rows.append({"figure": int(match[1]), "pdf_page": page_index + 1,
                                 "caption": text[match.start():], "caption_bbox": list(block[:4]),
                                 "preview": str(image.relative_to(ROOT))})
        inventory.append({"paper": path.parent.name, "figures": rows})
    (ROOT / "figure_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"papers": len(inventory), "captions": sum(len(p["figures"]) for p in inventory)}))


if __name__ == "__main__":
    main()
