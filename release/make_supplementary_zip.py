#!/usr/bin/env python3
"""CVPR 2027 부록 제출 파일 하나(ZIP)를 만든다: 부록 PDF + 익명 코드 정리본.

    python3 release/make_supplementary_zip.py [--pdf docs/paper/body_v2.7/cvpr2027_supplement.pdf]

출력: dist/cvpr2027_supplementary.zip
    supplementary.pdf
    code/README.md, code/nsd/, code/tools/, code/assets/   (release/cvpr2027_code/ 에서, verify/·__pycache__ 제외)

OpenReview 조건: ZIP 또는 PDF 한 개, 200 MB 이하.
묶기 전에 검사하고 하나라도 걸리면 ZIP을 만들지 않는다:
  * 코드·README·가중치 피클 안의 신원/경로/내부 이름 문자열, 한국어
  * PDF 저자 메타데이터, 첨부 파일, 내장되지 않은 글꼴, 본문 글자와 링크 주석(/URI)의 URL
    (허용 목록 ALLOWED_URLS의 주소와 그 하위 경로만 통과)
  * 전체 크기 200 MB
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "release/cvpr2027_code"
OUT = REPO / "dist/cvpr2027_supplementary.zip"
LIMIT = 200 * 1024 * 1024
STAMP = (2026, 11, 21, 0, 0, 0)
BANNED = re.compile(rb"qkrrlans|kimun|park\b|pkm|joms|aicps|cps2|dropbox|192\.168|/home/|/mnt/|/workspace|https?:|www\.|"
                    rb"github|pdv2|_2026|results/|scripts/|claude|anthropic|chatgpt|codex", re.I)
HANGUL = re.compile(r"[가-힣]")
# 익명성에 지장 없는 공개 자료(비교 방법 저장소, 인용 논문). 이 주소 자체와 하위 경로·판 번호만 허용한다.
ALLOWED_URLS = ("github.com/lus6-Jenny/RING", "github.com/BIT-MJY/SeqOT", "arxiv.org/abs/2010.10949")
URL = re.compile(r"(?:https?://|www\.)[^\s()<>\"']+", re.I)


def url_allowed(url: str) -> bool:
    """허용 주소와 같거나 그 뒤가 '/', '#', '?', 판 번호('v2')로 이어지면 허용한다."""
    u = re.sub(r"^(?:https?://)?(?:www\.)?", "", url.strip().rstrip(".,;:"), flags=re.I)
    for a in ALLOWED_URLS:
        host, _, path = a.partition("/")
        if u.lower().startswith(host.lower() + "/") and u[len(host) + 1:].startswith(path):
            rest = u[len(a):]
            if rest == "" or rest[0] in "/#?" or re.match(r"v\d+$", rest):
                return True
    return False


def pdf_uris(pdf: Path) -> set[str]:
    """링크 주석의 /URI 값(압축 스트림 안까지)."""
    raw = pdf.read_bytes()
    chunks = [raw]
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        try:
            chunks.append(zlib.decompress(m.group(1)))
        except zlib.error:
            pass
    return {m.group(1).decode("latin-1") for c in chunks for m in re.finditer(rb"/URI\s*\((.*?)\)", c, re.S)}


def code_files() -> list[Path]:
    files = [p for d in ("nsd", "tools", "assets") for p in sorted((CODE / d).rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts]
    return files + [CODE / "README.md"]


def scan_code(files: list[Path]) -> list[str]:
    problems = []
    for p in files:
        data = p.read_bytes()
        if p.suffix == ".pth":
            inner = zipfile.ZipFile(io.BytesIO(data))
            data = b"".join(inner.read(n) for n in inner.namelist() if n.endswith(".pkl"))
        elif p.suffix != ".npy" and HANGUL.search(data.decode("utf-8", "replace")):
            problems.append(f"{p.relative_to(CODE)}: Korean text")
        for m in URL.finditer(data.decode("latin-1")):
            if url_allowed(m.group()):
                data = data.replace(m.group().encode("latin-1"), b"")
        for m in BANNED.finditer(data):
            problems.append(f"{p.relative_to(CODE)}: {m.group().decode(errors='replace')!r}")
    return problems


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def scan_pdf(pdf: Path) -> list[str]:
    problems = []
    info = run("pdfinfo", str(pdf))
    author = re.search(r"^Author:[ \t]*(.*)$", info, re.M)
    if author and author.group(1).strip():
        problems.append(f"PDF author metadata: {author.group(1).strip()!r}")
    if run("pdfdetach", "-list", str(pdf)).strip().splitlines()[0:1] != ["0 embedded files"]:
        problems.append("PDF has embedded files")
    fonts = run("pdffonts", str(pdf)).splitlines()[2:]
    not_embedded = [f.split()[0] for f in fonts if len(f.split()) > 6 and f.split()[-5] != "yes"]
    if not_embedded:
        problems.append(f"PDF fonts not embedded: {not_embedded}")
    text = run("pdftotext", str(pdf), "-")
    urls = {m.group() for m in URL.finditer(text)} | pdf_uris(pdf)
    for url in sorted(urls):
        if not url_allowed(url):
            problems.append(f"PDF URL: {url}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=REPO / "docs/paper/body_v2.7/cvpr2027_supplement.pdf")
    args = ap.parse_args()

    files = code_files()
    problems = scan_code(files) + scan_pdf(args.pdf)
    if problems:
        print("ZIP을 만들지 않음:\n  " + "\n  ".join(problems))
        sys.exit(1)

    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        entries = [("supplementary.pdf", args.pdf)] + [("code/" + str(p.relative_to(CODE)), p) for p in files]
        for name, path in entries:
            zi = zipfile.ZipInfo(name, date_time=STAMP)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, path.read_bytes())
    size = tmp.stat().st_size
    if size > LIMIT:
        tmp.unlink()
        sys.exit(f"크기 {size} 바이트가 200 MB를 넘음")
    tmp.replace(OUT)
    pdf_sha = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    print(f"{OUT.relative_to(REPO)}  {size} 바이트, 파일 {len(files) + 1}개")
    print(f"  ZIP SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    print(f"  PDF {args.pdf.relative_to(REPO)} ({pdfinfo_pages(args.pdf)}쪽) SHA-256 {pdf_sha}")


def pdfinfo_pages(pdf: Path) -> str:
    m = re.search(r"^Pages:\s*(\d+)", run("pdfinfo", str(pdf)), re.M)
    return m.group(1) if m else "?"


if __name__ == "__main__":
    main()
