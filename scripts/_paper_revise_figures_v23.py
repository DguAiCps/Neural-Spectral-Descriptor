#!/usr/bin/env python3
"""v2.2 그림 1~3 원본 PPTX를 현재 v2.3 구조로 고쳐 PDF/PNG까지 다시 만든다.

원본은 docs/paper/body_v2.3/figs/original_v22/ 에 보존하고, 수정본은 같은 이름으로
figs/ 에 쓴다. 도형·글꼴·색은 원본 것을 그대로 복제해 쓰고, 바뀐 구조에 필요한
도형만 옮기거나 지우거나 추가한다. PDF는 Windows PowerPoint(COM)로 내보낸 뒤
Ghostscript로 원본과 같은 방식(pdfwrite)으로 다시 쓴다.
"""
from __future__ import annotations

import argparse
import copy
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from lxml import etree
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "docs/paper/body_v2.3/figs"
ORIGINAL = FIGS / "original_v22"
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
}
A, P, M, MC = (("{%s}" % NS[k]) for k in ("a", "p", "m", "mc"))

# 수학 기호 (Cambria Math, 굵은 이탤릭 소문자 = U+1D482 + 알파벳 순서)
BOLD = {c: chr(0x1D482 + ord(c) - ord("a")) for c in "abcdefghijklmnopqrstuvwxyz"}
ITAL = {c: chr(0x1D44E + ord(c) - ord("a")) for c in "abcdefghijklmnopqrstuvwxyz"}
ITAL["h"] = "ℎ"
ALPHA, ETA, TILDE, STAR = "\U0001D6FC", "\U0001D702", "̃", "⋆"


def xp(el, expr):
    """python-pptx 요소에서도 lxml 네임스페이스 XPath를 쓴다."""
    return etree._Element.xpath(el, expr, namespaces=NS)


# ----------------------------------------------------------------- 공통 도구
class Slide:
    """슬라이드 XML을 id로 다루는 얇은 도우미."""

    def __init__(self, prs: Presentation):
        self.prs = prs
        self.slide = prs.slides[0]
        self.tree = self.slide.shapes._spTree
        self.next_id = 4001

    def all(self, sid: int):
        return xp(self.tree, './/p:cNvPr[@id="%d"]' % sid)

    def el(self, sid: int):
        """mc:Choice 안의(수식이 있는) 도형을 우선 돌려준다."""
        hits = [h.getparent().getparent() for h in self.all(sid)]
        assert hits, sid
        for h in hits:
            if h.getparent().tag == MC + "Choice" or h.getparent().tag != MC + "Fallback":
                return h
        return hits[0]

    def top(self, sid: int):
        """spTree 바로 아래 요소(AlternateContent면 그것)를 돌려준다."""
        e = self.el(sid)
        while e.getparent() is not self.tree:
            e = e.getparent()
        return e

    def drop_fallback(self, sid: int) -> None:
        t = self.top(sid)
        if t.tag == MC + "AlternateContent":
            for fb in t.findall(MC + "Fallback"):
                t.remove(fb)

    def remove(self, *sids: int) -> None:
        for sid in sids:
            self.tree.remove(self.top(sid))

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def clone(self, sid: int, name: str):
        """도형을 복제해 spTree 끝에 붙이고 새 id를 준다(수식 대체 그림은 버린다)."""
        src = self.top(sid)
        dup = copy.deepcopy(src)
        if dup.tag == MC + "AlternateContent":
            for fb in dup.findall(MC + "Fallback"):
                dup.remove(fb)
        nid = self.new_id()
        for c in dup.iter(P + "cNvPr"):
            c.set("id", str(nid))
            c.set("name", name)
            for ext in c.findall(A + "extLst"):
                c.remove(ext)
        self.tree.append(dup)
        return nid

    # --- 위치 -----------------------------------------------------------
    def xfrm(self, sid: int):
        e = self.el(sid)
        pr = e.find(P + "spPr") if e.tag != P + "grpSp" else e.find(P + "grpSpPr")
        return pr.find(A + "xfrm")

    def move(self, sid: int, x=None, y=None, w=None, h=None, dx=0, dy=0, dw=0) -> None:
        for e in [h.getparent().getparent() for h in self.all(sid)]:
            pr = e.find(P + "spPr") if e.tag != P + "grpSp" else e.find(P + "grpSpPr")
            xf = pr.find(A + "xfrm")
            off, ext = xf.find(A + "off"), xf.find(A + "ext")
            off.set("x", str(int(x if x is not None else int(off.get("x")) + dx)))
            off.set("y", str(int(y if y is not None else int(off.get("y")) + dy)))
            if w is not None or dw:
                ext.set("cx", str(int(w if w is not None else int(ext.get("cx")) + dw)))
            if h is not None:
                ext.set("cy", str(int(h)))

    def box(self, sid: int):
        xf = self.xfrm(sid)
        off, ext = xf.find(A + "off"), xf.find(A + "ext")
        return int(off.get("x")), int(off.get("y")), int(ext.get("cx")), int(ext.get("cy"))

    # --- 글 -------------------------------------------------------------
    def set_lines(self, sid: int, lines, size=None) -> None:
        """단락을 다시 쓴다. lines 항목은 문자열 또는 [(글, 굵게), ...]."""
        for e in [h.getparent().getparent() for h in self.all(sid)]:
            body = e.find(P + "txBody")
            if body is None:
                continue
            paras = body.findall(A + "p")
            tmpl_p = paras[0]
            tmpl_r = tmpl_p.find(A + "r")
            assert tmpl_r is not None, sid
            end = tmpl_p.find(A + "endParaRPr")
            for ptag in paras:
                body.remove(ptag)
            for line in lines:
                runs = [(line, None)] if isinstance(line, str) else line
                p = etree.SubElement(body, A + "p")
                if tmpl_p.find(A + "pPr") is not None:
                    p.append(copy.deepcopy(tmpl_p.find(A + "pPr")))
                for text, bold in runs:
                    r = copy.deepcopy(tmpl_r)
                    rpr = r.find(A + "rPr")
                    if size:
                        rpr.set("sz", str(size))
                    if bold is not None:
                        rpr.set("b", "1" if bold else "0")
                    for key in ("err", "dirty"):
                        rpr.attrib.pop(key, None)
                    r.find(A + "t").text = text
                    p.append(r)
                if end is not None:
                    e2 = copy.deepcopy(end)
                    if size:
                        e2.set("sz", str(size))
                    p.append(e2)

    # --- 수식 -----------------------------------------------------------
    def math_runs(self, sid: int):
        return xp(self.el(sid), ".//m:r")

    def set_math_text(self, sid: int, mapping: dict) -> None:
        """m:t 글자를 바꾼다(예: {'𝒂': '𝒛'})."""
        self.drop_fallback(sid)
        for t in xp(self.el(sid), ".//m:t"):
            if t.text in mapping:
                t.text = mapping[t.text]

    def replace_texts(self, sid: int, mapping: dict) -> None:
        """글 run과 수식 run의 글자를 그대로 둔 서식 위에서 바꾼다."""
        self.drop_fallback(sid)
        for t in xp(self.el(sid), ".//a:t | .//m:t"):
            if t.text in mapping:
                t.text = mapping[t.text]

    def drop_subscript(self, sid: int) -> None:
        """sSub를 밑글자 없는 본체 run으로 바꾼다."""
        self.drop_fallback(sid)
        for ss in xp(self.el(sid), ".//m:sSub"):
            parent = ss.getparent()
            for r in list(ss.find(M + "e")):
                ss.addprevious(r)
            parent.remove(ss)

    def set_insets(self, sid: int, left: int, right: int) -> None:
        for body in xp(self.el(sid), ".//a:bodyPr"):
            body.set("lIns", str(left))
            body.set("rIns", str(right))

    def drop_superscript(self, sid: int, base_map: dict | None = None) -> None:
        """sSubSup → sSub (위첨자 ρ 제거)."""
        self.drop_fallback(sid)
        for ss in xp(self.el(sid), ".//m:sSubSup"):
            sub = etree.Element(M + "sSub")
            pr = etree.SubElement(sub, M + "sSubPr")
            ctrl = ss.find(M + "sSubSupPr").find(M + "ctrlPr")
            pr.append(copy.deepcopy(ctrl))
            sub.append(copy.deepcopy(ss.find(M + "e")))
            sub.append(copy.deepcopy(ss.find(M + "sub")))
            ss.getparent().replace(ss, sub)
        if base_map:
            self.set_math_text(sid, base_map)

    def add_tilde(self, sid: int, base_char: str) -> None:
        """base_char 글자를 가진 m:r을 m:acc(물결)로 감싼다."""
        self.drop_fallback(sid)
        for r in self.math_runs(sid):
            t = r.find(M + "t")
            if t.text != base_char:
                continue
            acc = etree.Element(M + "acc")
            accpr = etree.SubElement(acc, M + "accPr")
            chr_ = etree.SubElement(accpr, M + "chr")
            chr_.set(M + "val", TILDE)
            ctrl = etree.SubElement(accpr, M + "ctrlPr")
            ctrl.append(copy.deepcopy(r.find(A + "rPr")))
            e = etree.SubElement(acc, M + "e")
            parent = r.getparent()
            parent.replace(r, acc)
            e.append(r)

    def set_math_size(self, sid: int, size: int) -> None:
        for rpr in xp(self.el(sid), ".//a:rPr | .//a:endParaRPr"):
            rpr.set("sz", str(size))

    # --- 연결선 ---------------------------------------------------------
    def set_connector(self, sid: int, points, start=None, end=None) -> None:
        """꼭짓점 목록으로 연결선 기하를 정한다. start/end=(id, idx)."""
        e = self.el(sid)
        prst, rot, fh, fv, off, ext, adj = solve_geometry(points)
        sppr = e.find(P + "spPr")
        xf = sppr.find(A + "xfrm")
        for key in ("rot", "flipH", "flipV"):
            xf.attrib.pop(key, None)
        if rot:
            xf.set("rot", str(rot))
        if fh:
            xf.set("flipH", "1")
        if fv:
            xf.set("flipV", "1")
        xf.find(A + "off").set("x", str(off[0]))
        xf.find(A + "off").set("y", str(off[1]))
        xf.find(A + "ext").set("cx", str(ext[0]))
        xf.find(A + "ext").set("cy", str(ext[1]))
        geom = sppr.find(A + "prstGeom")
        geom.set("prst", prst)
        av = geom.find(A + "avLst")
        for gd in list(av):
            av.remove(gd)
        if adj is not None:
            gd = etree.SubElement(av, A + "gd")
            gd.set("name", "adj1")
            gd.set("fmla", "val %d" % adj)
        cpr = e.find(P + "nvCxnSpPr").find(P + "cNvCxnSpPr")
        for tag in ("stCxn", "endCxn"):
            for old in cpr.findall(A + tag):
                cpr.remove(old)
        for tag, ref in (("stCxn", start), ("endCxn", end)):
            if ref:
                c = etree.SubElement(cpr, A + tag)
                c.set("id", str(ref[0]))
                c.set("idx", str(ref[1]))

    def connector(self, template: int, name: str, points, start=None, end=None) -> int:
        nid = self.clone(template, name)
        self.set_connector(nid, points, start, end)
        return nid


def _path(prst, rot, fh, fv, off, ext, adj):
    """저장된 기하에서 슬라이드 좌표 꼭짓점을 계산한다."""
    a, b = ext
    if prst == "straightConnector1":
        pts = [(0, 0), (a, b)]
    elif prst == "bentConnector2":
        pts = [(0, 0), (a, 0), (a, b)]
    else:
        pts = [(0, 0), (adj * a, 0), (adj * a, b), (a, b)]
    if fh:
        pts = [(a - x, y) for x, y in pts]
    if fv:
        pts = [(x, b - y) for x, y in pts]
    th = math.radians(rot / 60000.0)
    cx, cy = a / 2.0, b / 2.0
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        out.append((off[0] + cx + dx * math.cos(th) - dy * math.sin(th),
                    off[1] + cy + dx * math.sin(th) + dy * math.cos(th)))
    return out


def solve_geometry(points):
    """원하는 꼭짓점을 만드는 (prst, rot, flipH, flipV, off, ext, adj)를 찾는다."""
    n = len(points)
    prst = {2: "straightConnector1", 3: "bentConnector2", 4: "bentConnector3"}[n]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    for rot in (0, 5400000, 10800000, 16200000):
        if rot in (0, 10800000):
            a, b, off = w, h, (x0, y0)
        else:
            a, b, off = h, w, (x0 + w / 2.0 - h / 2.0, y0 + h / 2.0 - w / 2.0)
        for fh in (0, 1):
            for fv in (0, 1):
                adj = None
                if n == 4:
                    p0 = _path(prst, rot, fh, fv, off, (a, b), 0.0)[1]
                    p1 = _path(prst, rot, fh, fv, off, (a, b), 1.0)[1]
                    want = points[1]
                    denom = (p1[0] - p0[0]) if abs(p1[0] - p0[0]) > 1 else (p1[1] - p0[1])
                    if abs(denom) < 1:
                        continue
                    num = (want[0] - p0[0]) if abs(p1[0] - p0[0]) > 1 else (want[1] - p0[1])
                    adj = num / denom
                    if not 0 <= adj <= 1:
                        continue
                got = _path(prst, rot, fh, fv, off, (a, b), adj if adj is not None else 0)
                if all(abs(g[0] - p[0]) <= 3 and abs(g[1] - p[1]) <= 3 for g, p in zip(got, points)):
                    return (prst, rot, fh, fv, (int(round(off[0])), int(round(off[1]))),
                            (int(round(a)), int(round(b))),
                            None if adj is None else int(round(adj * 100000)))
    raise ValueError("no geometry for %r" % (points,))


def make_math_run(template_r, text: str, bold: bool | None = None):
    r = copy.deepcopy(template_r)
    r.find(M + "t").text = text
    rpr = r.find(A + "rPr")
    if bold is not None:
        rpr.set("b", "1" if bold else "0")
    return r


# ------------------------------------------------------------------ 그림 2
def revise_overview(prs: Presentation) -> None:
    s = Slide(prs)
    # ---- Stage 1: 원통 블록 제거, a_i→z_i ---------------------------------
    s.remove(3295, 3302, 3280, 3303, 3297, 3374, 3319)
    s.set_connector(3304, [(4630757, 1032318), (6850824, 1032481)], (3307, 3), (3305, 1))
    s.set_math_text(3305, {BOLD["a"]: BOLD["z"]})
    s.set_math_text(3292, {BOLD["a"]: BOLD["z"]})

    # ---- Stage 2: 재선택·두 번째 통과·공유 가중치 제거 ---------------------
    s.remove(3333, 3334, 3335, 3336, 3338, 3339, 3340, 3274, 3345, 3268, 3310, 3329,
             3372, 46, 50, 53, 56, 3327, 3341)
    s.set_lines(3285, ["Stage 2: Temporal spectral difference"])
    s.set_lines(3369, ["Stage 3: Stored state"])

    D, C = 3331, 3332
    s.move(D, x=7560000, y=1698427, w=1190000, h=533698)
    s.set_lines(D, ["DiffAttnConv", "2 layers"], size=1800)
    s.move(C, x=9220000, y=1698427, w=2100000, h=533698)
    s.set_lines(C, ["Complex difference", "2 updates"], size=1800)
    # d_i → DiffAttnConv, d_j → DiffAttnConv(아래), DiffAttnConv → 복소 차이
    s.set_connector(3323, [(7228823, 1960379), (7560000, 1960379)], (3326, 3), (D, 1))
    s.set_connector(3337, [(6150563, 2914331), (8155000, 2914331), (8155000, 2232125)],
                    (3313, 3), (D, 2))
    s.connector(3323, "arrow attention", [(8750000, 1965276), (9220000, 1965276)], (D, 3), (C, 1))
    # z_j → 복소 차이(왼쪽 아래), z_i → 복소 차이(위), z_i → 크기 색인
    s.connector(3368, "arrow z_j", [(6150563, 3966442), (8850000, 3966442),
                                    (8850000, 2140000), (9220000, 2140000)], (3292, 3))
    s.set_connector(3368, [(7228824, 1032481), (12300000, 1032481)], (3305, 3), (3366, 1))
    s.connector(3323, "arrow z_i branch", [(10270000, 1032481), (10270000, 1698427)], None, (C, 0))
    # 점프 호를 새 교차점으로
    s.move(3365, x=10270000 - 48780, y=1530000 - 48780)
    # 주의 가중치·세기 표시 (k_ij, v_ij 꼴 복제)
    lab = s.clone(3361, "label alpha eta")
    s.move(lab, x=8770000, y=1700000, w=450000)
    s.set_math_size(lab, 1600)
    runs = s.math_runs(lab)
    texts = [ALPHA, ITAL["i"] + ITAL["j"], ", ", ETA, ITAL["i"]]
    assert len(runs) == len(texts), [r.find(M + "t").text for r in runs]
    for r, t in zip(runs, texts):
        r.find(M + "t").text = t
        if t in (ALPHA, ETA):
            r.find(A + "rPr").set("b", "0")

    # ---- Stage 2 inset: DiffAttnConv 층 설명, 주의 가중치 출력 -------------
    s.move(3347, x=8950000, w=3370000)
    for sid in (3348, 3349, 3351, 3350, 3352, 3353, 3357):
        s.move(sid, dx=260000)
    for sid in (3356, 3358, 3359):
        s.move(sid, dx=260000, dw=-260000)
    s.move(3361, x=10357318, y=3045965)
    s.move(3362, x=10333530)
    s.move(3354, x=8995000, w=3300000)
    s.set_lines(3354, [[("DiffAttnConv ", True), ("– 2 layers, one pass", False)]])
    s.move(2, y=2715000)
    s.move(3364, y=3060000 - 200400)
    s.set_connector(3363, [(11415183, 3060000), (11684859, 3060000)], (3355, 0), (3364, 1))
    att = s.clone(2, "label attention")
    s.move(att, y=3330000)
    s.set_lines(att, ["Attention"])
    sym = s.clone(3364, "symbol alpha")
    s.move(sym, y=3690000 - 200400)
    runs = s.math_runs(sym)
    runs[0].find(M + "t").text = ALPHA
    runs[0].find(A + "rPr").set("b", "0")
    runs[1].find(M + "t").text = ITAL["i"] + ITAL["j"]
    s.connector(3363, "arrow attention out", [(11415183, 3690000), (11684859, 3690000)],
                (3355, 0), (sym, 1))
    s.set_connector(60, [(8950000, 3189930), (8400000, 3189930), (8400000, 2232125)], (3347, 1))

    # ---- Stage 3: 크기 색인·재계량 키·수정 계수의 저장 상태 ----------------
    Mb, Rb = 3366, 3367
    s.move(Mb, x=12300000, y=782481, w=1060000, h=500000)
    s.set_lines(Mb, ["Magnitude", "index"], size=1800)
    s.move(Rb, x=12300000, y=1360000, w=1060000, h=340000)
    s.set_lines(Rb, ["Re-metrize"], size=1800)
    s.set_connector(3330, [(7039824, 1770303), (7039824, 1530000), (12300000, 1530000)],
                    (3326, 0), (Rb, 1))
    s.move(3378, x=13573953, y=1453953)
    s.move(3382, x=13573953, y=2373953)
    s.connector(3323, "arrow m_i", [(13360000, 1032481), (13650000, 1032481), (13650000, 1453953)],
                (Mb, 3), (3379, 0))
    s.connector(3323, "arrow r_i", [(13360000, 1530000), (13573953, 1530000)], (Rb, 3), (3379, 2))
    s.connector(3323, "arrow concat", [(13650000, 1606047), (13650000, 2373953)], (3379, 4), (3383, 0))
    zt = s.clone(3305, "symbol z tilde")
    s.move(zt, x=12420000, y=1874209)
    s.add_tilde(zt, BOLD["z"])
    s.connector(3323, "arrow z tilde", [(11320000, 1965276), (12420000, 1965276)], (C, 3), (zt, 1))
    s.connector(3323, "arrow z tilde concat", [(12798000, 1965276), (13250000, 1965276),
                                               (13250000, 2450000), (13573953, 2450000)], (zt, 3), (3383, 2))
    s.move(3322, x=12150000, y=2130000, w=1060000)
    s.set_lines(3322, ["Modified", "spectra"])
    s.connector(3323, "arrow state", [(13650000, 2526047), (13650000, 2780000)], (3383, 4), (3306, 0))
    s.move(3306, x=13400000, y=2780000, w=500000)
    s.drop_superscript(3306)
    s.remove(3299)
    for name, x, y, base in (("symbol m_i", 13660000, 1130000, BOLD["m"]),
                             ("symbol r_i", 13330000, 1610000, BOLD["r"])):
        sid = s.clone(3364, name)
        s.move(sid, x=x, y=y, w=300000, h=300000)
        s.math_runs(sid)[0].find(M + "t").text = base


# ------------------------------------------------------------------ 그림 3
def revise_matching(prs: Presentation) -> None:
    prs.slide_height = 4150000
    s = Slide(prs)
    DY = 200000
    s.drop_superscript(212)
    s.set_math_text(214, {BOLD["f"]: BOLD["r"]})
    s.drop_superscript(215, {BOLD["a"]: BOLD["m"]})
    s.move(215, y=701043, h=182134)
    s.remove(232, 233)
    s.set_connector(221, [(1299043, 792110), (2105417, 792632)], (215, 3), (222, 1))
    zq = s.clone(214, "symbol z tilde q")
    s.move(zq, x=958529, y=1050000)
    s.set_math_text(zq, {BOLD["r"]: BOLD["z"]})
    s.add_tilde(zq, BOLD["z"])
    s.connector(217, "arrow z tilde q", [(443487, 789392), (756055, 789392),
                                         (756055, 1141067), (958529, 1141067)], (212, 3), (zq, 1))
    for sid in (236, 234, 235, 237, 238, 242):
        s.move(sid, dy=DY)
    s.set_connector(246, [(1128786, 1232134), (1125786, 1329372 + DY)], (zq, 2), (236, 0))
    s.set_connector(239, [(1125786, 1920633 + DY), (1125786, 2177707 + DY), (2115677, 2177707 + DY)],
                    (236, 2), (238, 1))
    s.set_connector(240, [(5036925, 1920633 + DY), (5036925, 2177707 + DY), (3964517, 2177707 + DY)],
                    (237, 2), (238, 3))
    s.set_connector(244, [(5036925, 686208), (5036925, 1136900 + DY), (3040097, 1136900 + DY),
                          (3040097, 1438797 + DY)], (224, 2), (234, 0))
    s.set_connector(245, [(5036925, 686208), (5036925, 1329373 + DY)], (224, 2), (237, 0))
    s.set_connector(247, [(2373732, 1629655 + DY), (1792151, 1625002 + DY)], (234, 1), (236, 3))
    for sid in (230, 231):
        s.set_lines(sid, ["top-100"])
        s.move(sid, x=3700000, w=780000)
    s.set_connector(227, [(3708146, 351750), (4500000, 351750), (4500000, 513191), (4866668, 513191)],
                    (211, 3), (224, 1))
    s.set_connector(228, [(3708146, 792632), (4500000, 792632), (4500000, 513191), (4866668, 513191)],
                    (222, 3), (224, 1))
    s.drop_superscript(224)
    s.drop_superscript(235)
    s.drop_superscript(234, {BOLD["a"]: BOLD["z"]})
    s.add_tilde(234, BOLD["z"])
    # 최종 매치 → 반환 장소 j*
    s.move(242, x=1940097, w=2200000)
    s.drop_fallback(242)
    e = s.el(242)
    runs = xp(e, ".//a:r")
    runs[0].find(A + "t").text = "Returned place "
    acc = xp(e, ".//m:acc")[0]
    tmpl = xp(acc, ".//m:r")[0]
    sup = etree.Element(M + "sSup")
    pr = etree.SubElement(sup, M + "sSupPr")
    pr.append(copy.deepcopy(acc.find(M + "accPr").find(M + "ctrlPr")))
    ee = etree.SubElement(sup, M + "e")
    ee.append(make_math_run(tmpl, ITAL["j"]))
    ss = etree.SubElement(sup, M + "sup")
    ss.append(make_math_run(tmpl, STAR))
    acc.getparent().replace(acc, sup)
    s.set_connector(243, [(3040097, 2365829 + DY), (3040097, 2502430 + DY)], (238, 2), (242, 0))
    # 시간 증거 → 수락 점수
    T = s.clone(238, "box temporal evidence")
    s.move(T, x=1940097, y=3202430, w=2200000)
    s.set_lines(T, ["Temporal evidence"])
    Ab = s.clone(242, "box acceptance")
    s.move(Ab, y=3702430)
    e = s.el(Ab)
    xp(e, ".//a:r")[0].find(A + "t").text = "Acceptance "
    om = xp(e, ".//m:oMath")[0]
    sup = om.find(M + "sSup")
    tmpl = xp(sup, ".//m:r")[0]
    for t in ("\U0001D446", "(", ITAL["q"], ", "):
        sup.addprevious(make_math_run(tmpl, t))
    sup.addnext(make_math_run(tmpl, ")"))
    s.connector(243, "arrow to temporal", [(3040097, 2702430 + 376244), (3040097, 3202430)], (242, 2), (T, 0))
    s.connector(243, "arrow to acceptance", [(3040097, 3202430 + 376244), (3040097, 3702430)], (T, 2), (Ab, 0))


# ------------------------------------------------------------------ 그림 1
def revise_intro(prs: Presentation) -> None:
    """패널 b만: 문맥 코드 c·검색키 f 경로를 DiffAttnConv가 복소 스펙트럼을 고치는 경로로 바꾼다."""
    s = Slide(prs)
    s.set_lines(1199, ["DiffAttnConv"], size=1800)
    s.set_insets(1199, 0, 0)
    s.replace_texts(1230, {"Invariant": "Complex", "spectral key ": "spectrum ", BOLD["d"]: BOLD["z"]})
    for sid in (1219, 1167):
        s.replace_texts(sid, {"Trajectory-context code ": "Temporal attention ", BOLD["c"]: ALPHA})
    s.replace_texts(1235, {"Complex DFT features ": "Invariant key ", BOLD["a"]: BOLD["d"]})
    s.drop_subscript(1235)
    s.replace_texts(1238, {"for matching": "for retrieval"})
    for sid in (1229, 1169):
        s.replace_texts(sid, {"Retrieval key  ": "Modified  ", BOLD["f"]: BOLD["z"]})
        s.add_tilde(sid, BOLD["z"])
    for sid in (1188, 1189):
        s.replace_texts(sid, {BOLD["f"]: BOLD["z"]})
        s.add_tilde(sid, BOLD["z"])
    s.replace_texts(1190, {"retrieval-key space": "modified-spectrum space"})
    s.set_math_size(1190, 1400)
    s.move(1190, x=5710484)


# ------------------------------------------------------------- 내보내기
def export_pdf(pptx: Path, pdf: Path) -> None:
    """Windows PowerPoint로 PDF를 만들고 Ghostscript로 원본과 같은 형식으로 다시 쓴다."""
    powershell = shutil.which("powershell.exe")
    assert powershell, "Windows PowerPoint가 필요하다"
    temp_win = subprocess.check_output([powershell, "-NoProfile", "-NonInteractive",
                                        "-Command", "$env:TEMP"], text=True).strip()
    temp_linux = Path(subprocess.check_output(["wslpath", "-u", temp_win], text=True).strip())
    with tempfile.TemporaryDirectory(prefix="nsd_fig_", dir=temp_linux) as tmp:
        folder = Path(tmp)
        deck, raw = folder / pptx.name, folder / "raw.pdf"
        shutil.copyfile(pptx, deck)
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
        win = lambda p: subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()
        subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-File", win(script), "-DeckPath", win(deck), "-PdfPath", win(raw)],
                       check=True, timeout=180)
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
                        "-dPDFSETTINGS=/prepress", "-dCompatibilityLevel=1.5",
                        "-sOutputFile=" + str(pdf), str(raw)], check=True, timeout=120)
    subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "240", str(pdf),
                    str(pdf.with_suffix(""))], check=True, timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-export", action="store_true", help="PPTX만 쓴다")
    parser.add_argument("--out", type=Path, default=FIGS)
    parser.add_argument("--only", nargs="*", default=None, help="처리할 그림 이름")
    args = parser.parse_args()
    jobs = (("fig_intro", revise_intro), ("fig_overview", revise_overview),
            ("fig_matching", revise_matching))
    for name, revise in jobs:
        if args.only and name not in args.only:
            continue
        source = ORIGINAL / (name + ".pptx")
        prs = Presentation(str(source))
        revise(prs)
        target = args.out / (name + ".pptx")
        prs.save(str(target))
        print("wrote", target)
        if not args.no_export:
            export_pdf(target, target.with_suffix(".pdf"))
            print("wrote", target.with_suffix(".pdf"), target.with_suffix(".png"))


if __name__ == "__main__":
    main()
