#!/usr/bin/env python3
"""실측 JSON으로 편집 가능한 Figure 4 PPTX와 PowerPoint PDF를 만든다.

의존성: python-pptx. 렌더에는 Windows PowerPoint와 pdftoppm이 필요하다.
예: /tmp/nsd_figure4_env/bin/python scripts/_paper_figure4.py --render
비용 결과가 없으면 좌표나 점을 지어내지 않고 미측정 패널을 만든다.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results/pdv2_table3_20261029/summary.json"
NSD_AP = ROOT / "results/pdv2_spectral_whitening_20261026/fixed_test/table2_summary.json"
BASELINE_AP = ROOT / "results/pdv2_baseline_seeds/summary.json"
DEFAULT_OUTPUT = ROOT / "docs/paper/body_v2.2/figs/fig_efficiency"
FONT = "Calibri"
INK = "0E2841"
GRID = "E8E8E8"
GRAY = "626C72"
STYLES = {
    "nsd": ("NSD (ours)", "196B24", "D9F2D0"),
    "bevplace": ("BEVPlace++", "753A85", "F0E1FF"),
    "ringsharp": ("RING#-L", "156082", "DCEAF7"),
    "scpp": ("SC++", "B34E19", "FBE2D5"),
    "seqot": ("SeqOT", "94791C", "FFFFA3"),
    "ring": ("RING", "27656A", "C1E5F5"),
    "ringpp": ("RING++", "687335", "E2E8CE"),
    "deepring": ("DeepRING", "894949", "FFB3B3"),
    "disco": ("DiSCO", "3B5666", "D4DEE5"),
    "helios": ("HeLiOS", "7B586F", "E8D9E3"),
    "imlpr": ("ImLPR", "74604E", "E5DCCE"),
}
MARKERS = {1000: MSO_SHAPE.OVAL, 5000: MSO_SHAPE.RECTANGLE,
           10000: MSO_SHAPE.ISOSCELES_TRIANGLE}


def sha256(path: Path) -> str:
    """파일의 원천 확인용 SHA256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    """UTF-8 JSON 객체를 읽는다."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_points(summary: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """통과한 실측만 Table 2 AP와 결합하며 미측정 행은 그리지 않는다."""
    main = read_json(NSD_AP)
    baselines = read_json(BASELINE_AP)
    accuracy = {key: val["mean_std"]["AP"]
                for key, val in baselines["methods"].items()}
    accuracy["nsd"] = main["mean_std"]["whitening25"]["AP"]
    payload = read_json(summary) if summary.exists() else {}
    if payload and payload.get("schema") != "pdv2-table3-v1":
        raise ValueError("알 수 없는 표3 스키마: 비용을 추정해 그리지 않는다")
    points = []
    seen = set()
    for row in payload.get("measurements", []):
        if row.get("status") != "passed":
            continue
        if row.get("publish_ready") is False:
            continue
        method = row["method"]
        n_map = int(row["n_map"])
        if method not in accuracy or method not in STYLES:
            raise ValueError("AP 원천 또는 스타일이 없는 방법: " + method)
        if n_map not in MARKERS:
            raise ValueError("정의하지 않은 지도 크기: " + str(n_map))
        if (method, n_map) in seen:
            raise ValueError("방법·지도 크기 실측이 중복된다")
        seen.add((method, n_map))
        raw_memory = row["resident_bytes"].get("total")
        memory = (float(raw_memory) / 1e6 if raw_memory is not None and row.get("memory_publish_ready", True) and row["resident_bytes"].get("publish_ready", True) else None)
        elapsed = float(row["timings_ms"]["processing"]["mean"])
        ap = float(accuracy[method]["mean"]) * 100
        ap_sd = float(accuracy[method].get("std") or 0) * 100
        if not all(math.isfinite(v) for v in (elapsed, ap, ap_sd) + (() if memory is None else (memory,))):
            raise ValueError("유한하지 않은 실측을 그리지 않는다")
        if (memory is not None and memory <= 0) or elapsed <= 0 or not 0 <= ap <= 100 or ap_sd < 0:
            raise ValueError("비용·AP·표준편차 범위 오류")
        points.append({"method": method, "n_map": n_map,
                       "memory_mb": memory, "processing_ms": elapsed,
                       "ap_percent": ap, "ap_sd_percent": ap_sd})
    points.sort(key=lambda x: (list(STYLES).index(x["method"]), x["n_map"]))
    return points, payload.get("protocol", {})


def text_box(slide: Any, name: str, x: float, y: float, w: float,
             h: float, value: str, size: float = 16, color: str = INK,
             bold: bool = False, align: Any = PP_ALIGN.CENTER,
             rotation: float = 0) -> Any:
    """편집 가능한 Calibri 텍스트 상자를 추가한다."""
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = name
    shape.rotation = rotation
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_before = p.space_after = Pt(0)
        for run in p.runs:
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = RGBColor.from_string(color)
    return shape


def line(slide: Any, name: str, x1: float, y1: float, x2: float,
         y2: float, color: str = INK, width: float = 1) -> Any:
    """편집 가능한 직선을 추가한다."""
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),
                                       Inches(y1), Inches(x2), Inches(y2))
    shape.name = name
    shape.line.color.rgb = RGBColor.from_string(color)
    shape.line.width = Pt(width)
    return shape


def marker(slide: Any, name: str, x: float, y: float, n_map: int,
           outline: str, fill: str, size: float = 0.105) -> None:
    """지도 크기를 나타내는 편집 가능한 마커를 추가한다."""
    shape = slide.shapes.add_shape(MARKERS[n_map], Inches(x - size / 2),
                                   Inches(y - size / 2), Inches(size), Inches(size))
    shape.name = name
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.color.rgb = RGBColor.from_string(outline)
    shape.line.width = Pt(1.25)


def log_ticks(values: List[float]) -> Tuple[float, float, List[float]]:
    """실측의 양수 범위를 덮는 1·2·5 로그 눈금을 만든다."""
    lo = math.log10(min(values)) - 0.12
    hi = math.log10(max(values)) + 0.12
    if hi - lo < 0.5:
        midpoint = (hi + lo) / 2
        lo, hi = midpoint - 0.25, midpoint + 0.25
    ticks = [m * 10 ** power
             for power in range(math.floor(lo) - 1, math.ceil(hi) + 1)
             for m in (1, 2, 5)
             if lo <= math.log10(m * 10 ** power) <= hi]
    if len(ticks) > 5:
        ticks = ticks[::2]
    return lo, hi, ticks


def draw_panel(slide: Any, points: List[Dict[str, Any]], panel: int,
               field: str, title: str, xlabel: str, top: float,
               ap_limits: Tuple[float, float]) -> None:
    """AP와 실측 비용을 선형 Y·로그 X의 네이티브 도형으로 그린다."""
    points = [point for point in points if point.get(field) is not None]
    left, right = ((0.59, 3.37) if panel == 0 else (4.13, 6.91))
    bottom = top + 1.72
    prefix = "panel_" + str(panel + 1)
    text_box(slide, prefix + "_title", left - 0.10, top - 0.39,
             right - left + 0.20, 0.28, title, size=18, bold=True)
    ymin, ymax = ap_limits
    ystep = 5 if ymax - ymin <= 25 else 10
    if ymax - ymin > 60:
        ystep = 20
    yticks = list(range(math.ceil(ymin / ystep) * ystep,
                        math.floor(ymax / ystep) * ystep + 1, ystep))
    for tick in yticks:
        y = bottom - (tick - ymin) / (ymax - ymin) * (bottom - top)
        line(slide, prefix + "_ygrid_" + str(tick), left, y, right, y, GRID, 0.6)
        text_box(slide, prefix + "_ytick_" + str(tick), left - 0.42,
                 y - 0.12, 0.34, 0.24, str(tick), size=14,
                 align=PP_ALIGN.RIGHT)
    line(slide, prefix + "_xaxis", left, bottom, right, bottom, width=1.25)
    line(slide, prefix + "_yaxis", left, top, left, bottom, width=1.25)
    text_box(slide, prefix + "_ylabel", left - 0.81, top + 0.55,
             0.75, 0.28, "AP (%)", size=16, rotation=270)
    text_box(slide, prefix + "_xlabel", left - 0.07, bottom + 0.36,
             right - left + 0.14, 0.26, xlabel, size=16)
    if not points:
        text_box(slide, prefix + "_pending", left + 0.09, top + 0.57,
                 right - left - 0.18, 0.26, "Measurements pending", size=16,
                 color=GRAY)
        text_box(slide, prefix + "_empty", left + 0.09, top + 0.90,
                 right - left - 0.18, 0.25, "No cost values plotted", size=14,
                 color=GRAY)
        return
    lo, hi, ticks = log_ticks([point[field] for point in points])

    def map_x(value: float) -> float:
        return left + (math.log10(value) - lo) / (hi - lo) * (right - left)

    def map_y(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    for tick in ticks:
        x = map_x(tick)
        line(slide, prefix + "_xtickmark_" + str(tick), x, bottom,
             x, bottom + 0.055, width=0.9)
        text_box(slide, prefix + "_xtick_" + str(tick), x - 0.30,
                 bottom + 0.10, 0.60, 0.21, format(tick, ".3g"), size=14)
    for point in points:
        method = point["method"]
        _, outline, fill = STYLES[method]
        x, y = map_x(point[field]), map_y(point["ap_percent"])
        name = prefix + "_" + method + "_n" + str(point["n_map"])
        sd = point["ap_sd_percent"]
        if sd:
            ylo = map_y(point["ap_percent"] - sd)
            yhi = map_y(point["ap_percent"] + sd)
            line(slide, name + "_sd", x, ylo, x, yhi, outline, 1)
            line(slide, name + "_sd_low", x - 0.035, ylo,
                 x + 0.035, ylo, outline, 1)
            line(slide, name + "_sd_high", x - 0.035, yhi,
                 x + 0.035, yhi, outline, 1)
        marker(slide, name, x, y, point["n_map"], outline, fill)


def build_pptx(points: List[Dict[str, Any]], output: Path,
               provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Fig. 1–3 팔레트와 글꼴을 쓰는 단일 슬라이드를 만든다."""
    methods = [key for key in STYLES if key in {p["method"] for p in points}]
    if not methods:
        methods = ["nsd", "bevplace"]
    legend_rows = math.ceil(len(methods) / 3)
    top = 0.79 + (legend_rows - 1) * 0.28
    prs = Presentation()
    prs.slide_width = Inches(7.2)
    prs.slide_height = Inches(top + 2.74)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for idx, method in enumerate(methods):
        row, column = divmod(idx, 3)
        # 방법 수가 둘이면 중앙에 놓고 이후에는 세 열로 확장한다.
        if len(methods) == 1:
            x = 2.87
        elif len(methods) == 2:
            x = [1.64, 4.02][column]
        else:
            x = 0.62 + column * 2.22
        y = 0.15 + row * 0.28
        label, outline, fill = STYLES[method]
        marker(slide, "legend_" + method, x, y + 0.10, 1000,
               outline, fill, 0.115)
        text_box(slide, "legend_label_" + method, x + 0.13, y - 0.03,
                 1.95, 0.26, label, size=16, bold=method == "nsd",
                 align=PP_ALIGN.LEFT)
    ap_values = [p["ap_percent"] - p["ap_sd_percent"] for p in points]
    ap_maxima = [p["ap_percent"] + p["ap_sd_percent"] for p in points]
    ymin = max(0, math.floor((min(ap_values) - 3) / 5) * 5) if points else 80
    ymax = min(100, math.ceil((max(ap_maxima) + 3) / 5) * 5) if points else 95
    draw_panel(slide, points, 0, "memory_mb", "a) Resident state",
               "Memory (MB, log scale)", top, (ymin, ymax))
    draw_panel(slide, points, 1, "processing_ms", "b) Processing time",
               "Time (ms, log scale)", top, (ymin, ymax))
    y = top + 2.57
    text_box(slide, "map_legend_heading", 1.23, y - 0.08,
             1.55, 0.24, "Map keyframes:", size=14, align=PP_ALIGN.RIGHT)
    for idx, n_map in enumerate(MARKERS):
        x = 2.98 + idx * 0.84
        marker(slide, "map_legend_" + str(n_map), x, y + 0.035,
               n_map, INK, "FFFFFF", 0.105)
        text_box(slide, "map_label_" + str(n_map), x + 0.10, y - 0.08,
                 0.55, 0.24, str(n_map // 1000) + "k", size=14,
                 align=PP_ALIGN.LEFT)
    slide.notes_slide.notes_text_frame.text = (
        "Figure 4. Accuracy versus measured resource use.\n"
        "All elements are editable PowerPoint shapes/text, not a raster plot.\n"
        "AP: Table 2 six-city macro mean over 45 positive sequences from the "
        "46-sequence test split; bars show seed SD (three seeds for learned methods).\n"
        "Costs: seed 1 for learned methods, matched hardware/workload subset; unique live CPU numerical "
        "buffers plus GPU allocator-active memory (torch.cuda.memory_allocated) "
        "and separately traced persistent native requested buffers. "
        "GPU model tensors are not added again. CPU allocated array capacity "
        "is included; shared raw-input fixtures are excluded. GPU reserved/peak "
        "and process RSS are reported separately. Time includes "
        "raw encoding + incremental update + retrieval/reranking/final scoring.\n"
        "I/O, common KISS-ICP, and the 15-keyframe observation wait are excluded "
        "from processing time. See the measurement protocol for exact scope.\n"
        "No data point is drawn without a passed measured cost record.\n\n"
        + json.dumps(provenance, ensure_ascii=False, indent=2))
    prs.core_properties.title = "Figure 4 — Accuracy and resource use"
    prs.core_properties.author = "Kimun-Park"
    prs.core_properties.subject = "Editable scientific figure; measured costs only"
    # Office 기본 테마의 입체 그림자를 없애 Fig. 1–3의 평면 스타일을 따른다.
    for shape in slide.shapes:
        shape.shadow.inherit = False
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output.with_suffix(".pptx")))
    return {"width_inches": 7.2, "height_inches": top + 2.74,
            "native_shapes": len(slide.shapes), "raster_images": 0,
            "font": FONT, "methods": methods,
            "y_limits_percent": [ymin, ymax]}


def write_latex(output: Path, manifest: Dict[str, Any]) -> None:
    """완료 상태와 실제 범위에 맞는 그림 환경·캡션을 저장한다."""
    if manifest["state"] == "pending":
        caption = (
            r"\textbf{Accuracy and resource use.} Planned NSD and BEVPlace++ "
            "comparison; cost measurements are pending.")
    else:
        protocol = manifest["protocol"]
        sequence = str(protocol.get("sequence", "matched workload")).split("/")[-1]
        sequence = sequence.replace("_", r"\_").replace("%", r"\%")
        caption = (
            r"\textbf{Accuracy and resource use.} Full-test six-city AP versus "
            + sequence + " resource costs. Bars: training-seed SD; markers: "
            "initial gallery size. Cost measurements use seed 1.")
        if manifest["state"] != "complete_all":
            caption += " Only completed measurements are shown."
    content = (
        "% Generated by scripts/_paper_figure4.py; edit the builder, not this file.\n"
        "\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\columnwidth]{figs/fig_efficiency.pdf}\n"
        "\\caption{\\cmedit{" + caption + "}}\n"
        "\\label{fig:accuracy_efficiency}\n\\end{figure}\n")
    output.with_suffix(".tex").write_text(content, encoding="utf-8")


def render_powerpoint(output: Path) -> None:
    """자체 덱만 열고 닫아 PowerPoint PDF를 만들고 PNG로 렌더한다."""
    powershell = shutil.which("powershell.exe")
    if not powershell:
        raise RuntimeError("--render에는 Windows PowerPoint가 필요하다")
    temp_win = subprocess.check_output(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", "$env:TEMP"],
        text=True).strip()
    temp_linux = Path(subprocess.check_output(["wslpath", "-u", temp_win],
                                              text=True).strip())
    with tempfile.TemporaryDirectory(prefix="nsd_figure4_", dir=temp_linux) as tmp:
        folder = Path(tmp)
        deck = folder / "fig_efficiency.pptx"
        pdf = folder / "fig_efficiency.pdf"
        shutil.copyfile(output.with_suffix(".pptx"), deck)
        win_deck = subprocess.check_output(["wslpath", "-w", str(deck)],
                                           text=True).strip()
        win_pdf = subprocess.check_output(["wslpath", "-w", str(pdf)],
                                          text=True).strip()
        script = folder / "export.ps1"
        script.write_text(
            "param([string]$DeckPath, [string]$PdfPath)\n"
            "$ErrorActionPreference = 'Stop'\n"
            "$wasRunning = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0\n"
            "$app = $null\n$deck = $null\n"
            "try {\n"
            "  $app = New-Object -ComObject PowerPoint.Application\n"
            "  $deck = $app.Presentations.Open($DeckPath, $true, $false, $false)\n"
            "  $deck.SaveAs($PdfPath, 32)\n"
            "  $deck.Close()\n  $deck = $null\n"
            "} finally {\n"
            "  if ($null -ne $deck) { $deck.Close() }\n"
            "  if ($null -ne $app) {\n"
            "    if ((-not $wasRunning) -and ($app.Presentations.Count -eq 0)) { $app.Quit() }\n"
            "    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)\n"
            "  }\n}\n", encoding="utf-8")
        win_script = subprocess.check_output(["wslpath", "-w", str(script)],
                                             text=True).strip()
        subprocess.run([powershell, "-NoProfile", "-NonInteractive",
                        "-ExecutionPolicy", "Bypass", "-File", win_script,
                        "-DeckPath", win_deck, "-PdfPath", win_pdf], check=True,
                       timeout=120)
        shutil.copyfile(pdf, output.with_suffix(".pdf"))
    subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "240",
                    str(output.with_suffix(".pdf")), str(output)],
                   check=True, timeout=60)


def main() -> None:
    """입력을 검사하고 도표와 재생성 근거를 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    points, protocol = load_points(args.summary)
    sources = [NSD_AP, BASELINE_AP, Path(__file__).resolve()]
    if args.summary.exists():
        sources.append(args.summary)
    expected = [(m, n) for m in ("nsd", "bevplace") for n in MARKERS]
    expected_all = [(m, n) for m in STYLES for n in MARKERS]
    present = {(p["method"], p["n_map"]) for p in points}
    complete_all = all(x in present for x in expected_all) and all(
        point["memory_mb"] is not None for point in points)
    state = "complete_all" if complete_all else "complete_primary" if all(x in present for x in expected) else (
        "partial" if points else "pending")
    manifest = {"schema": "nsd-figure4-v1", "state": state,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "sources": {str(p): sha256(p) for p in sources},
                "protocol": protocol, "points": points,
                "cost_scope": "matched workload subset; seed 1 for learned methods; not full-test mean",
                "ap_scope": "Table 2 six-city macro; three seeds for learned methods",
                "memory_unit": "MB = 1,000,000 bytes",
                "memory_quantity": "unique CPU numerical buffers + GPU allocator-active memory (torch.cuda.memory_allocated) + traced persistent native requested buffers",
                "memory_accounting": {
                    "cpu": "unique live numerical buffers including allocated array capacity",
                    "gpu": "actual torch.cuda.memory_allocated plus separately traced persistent native requested capacity; GPU model tensors are not added again; not physical driver allocation granularity",
                    "excluded": "shared preloaded raw-input fixture",
                    "reported_separately": ["CUDA reserved", "CUDA peak", "process RSS"],
                },
                "time_quantity": "timings_ms.processing.mean",
                "error_bars": "AP seed standard deviation; no cost error bars",
                "pending_primary": [{"method": m, "n_map": n} for m, n in expected
                                    if (m, n) not in present],
                "pending_all": [{"method": m, "n_map": n} for m, n in expected_all
                                if (m, n) not in present]}
    manifest["layout"] = build_pptx(points, args.output, manifest)
    write_latex(args.output, manifest)
    if args.render:
        render_powerpoint(args.output)
    manifest["outputs"] = {ext: {"path": str(args.output.with_suffix(ext)),
                                  "sha256": sha256(args.output.with_suffix(ext))}
                           for ext in ([".pptx", ".pdf", ".png", ".tex"] if args.render
                                       else [".pptx", ".tex"])}
    manifest_path = args.output.parent / (args.output.name + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"state": state, "plotted_cost_records": len(points),
                      "output": str(args.output), "rendered": args.render},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
