#!/usr/bin/env python3
"""CVPR 공식 공개본과 출처를 보존하고 그림 캡션을 읽기용으로 추출한다."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/related works/CVPR_2024-2026_NSD"
ORIGIN = "https://openaccess.thecvf.com"
SELECTED = {
    "LiSA:": "LiSA", "DiffLoc:": "DiffLoc", "TransLoc4D:": "TransLoc4D",
    "Optimal Transport Aggregation for Visual Place Recognition": "SALAD",
    "BoQ:": "BoQ", "CricaVPR:": "CricaVPR",
    "On the Estimation of Image-matching Uncertainty": "SUE",
    "ForestLPR:": "ForestLPR", "HOTFormerLoc:": "HOTFormerLoc",
    "LightLoc:": "LightLoc", "Multi-Modal Aerial-Ground Cross-View": "AGPlace",
    "C-LaV:": "C-LaV", "BEV-SLD:": "BEV-SLD", "TACO: Task-Aware Contrastive": "TACO",
    "LEADER:": "LEADER", "HypeVPR:": "HypeVPR", "EfficientVPR:": "EfficientVPR",
}


def fetch(url: str, target: Path) -> bytes:
    """출처 URL을 그대로 다운로드하며 기존 다운로드는 재사용한다."""
    if target.exists():
        return target.read_bytes()
    response = requests.get(url, timeout=(15, 90))
    response.raise_for_status()
    data = response.content
    if target.suffix == ".pdf" and not data.startswith(b"%PDF"):
        raise ValueError("PDF가 아닌 응답: " + url)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    return data


def paper(record: dict) -> dict:
    """논문별 공식 HTML, 본문, PDF 부록과 검증 메타데이터를 보존한다."""
    directory = OUT / "papers" / f"{record['year']}_{record['short']}"
    html = fetch(record["url"], directory / "official.html")
    soup = BeautifulSoup(html, "html.parser")
    record["files"] = []
    record["supplement_pdf_available"] = False
    record["other_supplements"] = []
    for anchor in soup.find_all("a", href=True):
        label = anchor.get_text(strip=True).strip("[]").lower()
        link = urljoin(record["url"], anchor["href"])
        if label == "supp" and not link.lower().endswith(".pdf"):
            record["other_supplements"].append({"url": link, "downloaded": False})
        if label not in ("pdf", "supp") or not link.lower().endswith(".pdf"):
            continue
        kind = "main" if label == "pdf" else "supplement"
        destination = directory / (kind + ".pdf")
        data = fetch(link, destination)
        info = subprocess.check_output(["pdfinfo", str(destination)], text=True)
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
        subprocess.run(["pdftotext", "-layout", str(destination), str(directory / (kind + ".txt"))], check=True)
        record["files"].append({"kind": kind, "url": link, "path": str(destination.relative_to(OUT)),
                                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "pages": pages})
        if kind == "supplement":
            record["supplement_pdf_available"] = True
    if not any(item["kind"] == "main" for item in record["files"]):
        raise ValueError("공식 본문 PDF 링크 없음: " + record["url"])
    bib = soup.select_one(".bibref")
    if bib:
        record["bibtex"] = bib.get_text("\n", strip=True)
    print(f"저장 {record['year']} {record['short']}: {len(record['files'])} PDFs", flush=True)
    return record


def main() -> None:
    """연도별 공식 목차를 검사하고 선별한 논문만 병렬 다운로드한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    records, indexes = [], []
    for year in (2024, 2025, 2026):
        url = f"{ORIGIN}/CVPR{year}?day=all"
        cached = Path(f"/tmp/nsd_cvpr_search/{year}.html")
        target = OUT / "sources" / f"CVPR{year}_index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists() and not target.exists():
            target.write_bytes(cached.read_bytes())
        data = fetch(url, target)
        soup = BeautifulSoup(data, "html.parser")
        entries = soup.select("dt.ptitle a")
        indexes.append({"year": year, "url": url, "entries": len(entries), "sha256": hashlib.sha256(data).hexdigest()})
        for anchor in entries:
            title = anchor.get_text(" ", strip=True)
            for prefix, short in SELECTED.items():
                if title.startswith(prefix):
                    records.append({"year": year, "short": short, "title": title,
                                    "url": urljoin(ORIGIN, anchor["href"])})
                    break
    assert len(records) == len(SELECTED), (len(records), len(SELECTED))
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(paper, record): record for record in records}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"paper": futures[future], "error": repr(exc)})
    manifest = {"date_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
                "scope": "CVPR 2024–2026 main conference; selected NSD-related papers, not an exhaustive systematic review",
                "indexes": indexes, "papers": sorted(results, key=lambda r: (r["year"], r["short"])), "errors": errors}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"downloaded_papers": len(results), "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
