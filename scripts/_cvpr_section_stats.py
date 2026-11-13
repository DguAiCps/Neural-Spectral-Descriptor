#!/usr/bin/env python3
"""How CVPR papers spend their main body (before References) across sections.

Text runs with font family, size and position come from poppler's
``pdftohtml -xml`` (host tool; no Python PDF library needed).

Per paper it measures, for the canonical buckets Abstract / Introduction /
Related Work / Method / Experiments / Conclusion:
  * prose words and prose lines (body-font text at body size; captions,
    tables, figure text, display equations, footnotes, line numbers excluded),
  * column span from the section heading to the next top-level heading in
    reading order (1 column = 0.5 page; floats inside the span are included),
and per paper: figures and tables in the body (unique caption numbers),
Introduction paragraphs, contribution bullets, Related Work position, body
length in pages (where References start) and the number of references.

Usage
  python3 scripts/_cvpr_section_stats.py --out DIR        # whole corpus + ours
  python3 scripts/_cvpr_section_stats.py --dump PDF       # segmentation dump
"""
import argparse
import csv
import glob
import html
import json
import os
import re
import subprocess
import sys
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_GLOB = os.path.join(REPO, 'docs/related works/CVPR_2024-2026_NSD/papers/*/main.pdf')
SAMPLE_DIR = '/mnt/d/NSD_datasets/nsd_results/cvpr_section_stats_20261111/pdfs'
OURS = os.path.join(REPO, 'docs/paper/body_v2.3/cvpr2027_submission.pdf')

ZOOM = 2.0
BUCKETS = ['Abstract', 'Introduction', 'Related Work', 'Method', 'Experiments', 'Conclusion']
MATH_STEMS = ('cm', 'msbm', 'msam', 'eufm', 'eusm', 'rsfs', 'stixmath', 'cambriamath', 'lmmath',
              'latinmodernmath', 'txmi', 'txsy', 'txex', 'pxmi', 'pxsy', 'mathematicalpi', 'symbol',
              'dsrom', 'bbm', 'wasy', 'lasy', 'stmary', 'esint', 'mnsymbol', 'fontawesome', 'pifont',
              'zapfding', 'dingbat', 'stixsiz', 'stixgeneral', 'stixnonuni', 'xits', 'asana')
BOLD_RE = re.compile(r'(Medi|Bold|Bd\b|Semibold|SemiBold|Black|Heavy|Demi|CMBX|cmbx)')
BULLETS = ('•', '◦', '▪', '∙', '●', '‣', '⋄', '■', '□', '★', '➢', '►', '▶', '✓', '·')
ENUM_RE = re.compile(r'^(\(?\d{1,2}\)|\d{1,2}\.|\(?[ivx]{1,4}\)|\([a-e]\))\s+\S')
CAPTION_RE = re.compile(r'^(Figure|Fig\.|Table|Tab\.)\s*(\d+)\s*[.:|]')
UNNUMBERED_TOP = re.compile(r'^(Abstract|References|Bibliography|Acknowledg(e)?ments?|Limitations?'
                            r'|Conclusions?|Broader Impacts?|Appendix|Supplementary.*|LLM Usage.*|.*Statement|Ethics.*|Reproducibility.*)\.?$',
                            re.I)


# ----------------------------------------------------------------------------------------------
# PDF -> text runs
# ----------------------------------------------------------------------------------------------
def font_stem(family):
    f = family.split('+', 1)[-1]
    f = re.split(r'[-,]', f)[0]
    f = re.sub(r'(PSMT|PS|MT)$', '', f)
    f = re.sub(r'(BoldItalic|BoldOblique|Bold|Italic|Oblique|Regular|Roman|Medium|Semibold|Light)$', '', f, flags=re.I)
    return f.lower()


def read_runs(pdf):
    xml = subprocess.run(['pdftohtml', '-xml', '-i', '-q', '-nodrm', '-stdout', '-zoom', str(ZOOM), pdf],
                         capture_output=True, check=True).stdout.decode('utf-8', 'replace')
    fonts = {}
    for m in re.finditer(r'<fontspec id="(\d+)" size="(-?[\d.]+)" family="([^"]*)" color="([^"]*)"', xml):
        fam = m.group(3)
        fonts[m.group(1)] = dict(size=float(m.group(2)) / ZOOM, family=fam, stem=font_stem(fam),
                                 bold=bool(BOLD_RE.search(fam)))
    pages, runs = [], []
    for pm in re.finditer(r'<page number="(\d+)"[^>]*height="([\d.]+)" width="([\d.]+)">(.*?)</page>', xml, re.S):
        pno = int(pm.group(1))
        pages.append(dict(page=pno, h=float(pm.group(2)) / ZOOM, w=float(pm.group(3)) / ZOOM))
        for tm in re.finditer(r'<text top="(-?[\d.]+)" left="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)" '
                              r'font="(\d+)">(.*?)</text>', pm.group(4), re.S):
            raw = tm.group(6)
            txt = html.unescape(re.sub(r'<[^>]+>', '', raw))
            if not txt.strip():
                continue
            f = fonts[tm.group(5)]
            top, left = float(tm.group(1)) / ZOOM, float(tm.group(2)) / ZOOM
            w, h = float(tm.group(3)) / ZOOM, float(tm.group(4)) / ZOOM
            W = float(pm.group(3)) / ZOOM
            if left + w < 0.085 * W or left > 0.915 * W:
                continue  # margin: review-copy line numbers, side labels
            runs.append(dict(page=pno, top=top, left=left, right=left + w, bottom=top + h, text=txt,
                             size=f['size'], stem=f['stem'], family=f['family'],
                             bold=f['bold'] or '<b>' in raw))
    return pages, runs


def is_math(stem, body_stem):
    return stem != body_stem and stem.startswith(MATH_STEMS)


# ----------------------------------------------------------------------------------------------
# runs -> lines (per page and column)
# ----------------------------------------------------------------------------------------------
def build_lines(pages, runs):
    # body font = (stem, size) with most characters among non-bold runs on pages 1-7 (always body;
    # long reference lists on later pages must not win)
    cnt = Counter()
    for r in runs:
        if not r['bold'] and r['page'] <= 7:
            cnt[(r['stem'], round(r['size'] * 2) / 2)] += len(r['text'].strip())
    (body_stem, body_size), _ = cnt.most_common(1)[0]
    width = {p['page']: p['w'] for p in pages}
    for r in runs:
        mid = width[r['page']] / 2
        r['col'] = 0 if r['right'] <= mid + 2 else (1 if r['left'] >= mid - 2 else 2)
        r['cls'] = ('body' if r['stem'] == body_stem and abs(r['size'] - body_size) <= 0.3
                    else 'math' if is_math(r['stem'], body_stem) else 'other')
    lines = []
    key = lambda r: (r['page'], r['col'], (r['top'] + r['bottom']) / 2)
    tol = 0.45 * body_size
    for r in sorted(runs, key=key):
        c = (r['top'] + r['bottom']) / 2
        L = lines[-1] if lines else None
        if L and L['page'] == r['page'] and L['col'] == r['col'] and abs(c - L['c']) < tol:
            L['runs'].append(r)
        else:
            lines.append(dict(page=r['page'], col=r['col'], c=c, runs=[r]))
    for L in lines:
        rs = sorted(L['runs'], key=lambda r: r['left'])
        parts, gaps = [rs[0]['text']], []
        for a, b in zip(rs, rs[1:]):
            g = b['left'] - a['right']
            gaps.append(g)
            parts.append((' ' if g > 0.2 * body_size else '') + b['text'])
        txt = re.sub(r'\s+', ' ', ''.join(parts)).strip()
        nch = Counter()
        for r in rs:
            nch[r['cls']] += len(re.sub(r'\s', '', r['text']))
        main = [r for r in rs if r['cls'] != 'math'] or rs
        sz = Counter()
        for r in main:
            sz[round(r['size'] * 2) / 2] += len(r['text'])
        letters = [r for r in rs if re.search(r'[A-Za-z]', r['text'])]
        L.update(text=txt, x0=rs[0]['left'], x1=max(r['right'] for r in rs),
                 top=min(r['top'] for r in rs), bottom=max(r['bottom'] for r in rs),
                 size=sz.most_common(1)[0][0], first_size=rs[0]['size'], nch=nch, gaps=gaps,
                 all_bold=bool(letters) and all(r['bold'] for r in letters),
                 stems={r['stem'] for r in letters})
        del L['runs']
    return lines, body_stem, body_size


def classify_lines(lines, body_stem, body_size):
    for L in lines:
        toks = L['text'].split()
        num = sum(bool(re.fullmatch(r'[\d.,%±()\-–+/×x*]+', t)) for t in toks)
        big_gaps = sum(g > 1.5 * body_size for g in L['gaps'])
        textual = L['nch']['body'] + L['nch']['other']
        # numeric-heavy prose (citation lists, intervals) keeps normal word spacing; table rows have
        # column gaps
        tablelike = ((len(toks) >= 2 and num / len(toks) >= 0.5 and any(g > 0.8 * body_size for g in L['gaps']))
                     or big_gaps >= 3 or (big_gaps >= 2 and L['nch']['math'] == 0))
        L['textual'] = (L['col'] in (0, 1) and bool(re.search(r'[A-Za-z]{2}', L['text']))
                        and textual > 0 and L['nch']['body'] / textual >= 0.6 and not tablelike)
        L['prose'] = L['textual'] and L['nch']['body'] >= 0.4 * sum(L['nch'].values())
    # math-heavy text lines (inline intervals, symbols) are prose when they fill the column like justified
    # text: start at the column's left edge and end at its right edge. Display equations are centred.
    for c in (0, 1):
        x0 = Counter(round(L['x0']) for L in lines if L['prose'] and L['col'] == c).most_common(1)[0][0]
        x1 = Counter(round(L['x1']) for L in lines if L['prose'] and L['col'] == c).most_common(1)[0][0]
        for L in lines:
            if (L['col'] == c and L['textual'] and not L['prose'] and L['nch']['body'] >= 8
                    and abs(L['x0'] - x0) <= 1.5 and abs(L['x1'] - x1) <= 1.5):
                L['prose'] = True


# ----------------------------------------------------------------------------------------------
# headings, sections, positions
# ----------------------------------------------------------------------------------------------
def find_headings(lines, body_stem, body_size):
    heads = []
    for i, L in enumerate(lines):
        if L['col'] not in (0, 1) or not L['all_bold'] or len(L['text'].split()) > 16:
            continue
        if L['stems'] - {body_stem} and not all(s.startswith(('cm', body_stem[:4])) for s in L['stems']):
            continue
        t = L['text']
        big = max(L['size'], L['first_size']) >= body_size + 1.2  # first run: small-caps titles
        m = re.match(r'^(\d+)\.?\s+([A-Z].*)$', t)
        ms = re.match(r'^(\d+)\.(\d+)(\.\d+)?\.?\s+(\S.*)$', t)
        ma = re.match(r'^([A-H])\.?\s+([A-Z].*)$', t)
        if big and m:
            heads.append(dict(i=i, level=1, num=int(m.group(1)), title=m.group(2).strip()))
        elif big and UNNUMBERED_TOP.match(t):
            heads.append(dict(i=i, level=1, num=None, title=t.rstrip('.')))
        elif big and ma:
            heads.append(dict(i=i, level=1, num=ma.group(1), title=ma.group(2).strip()))
        elif ms and abs(L['size'] - body_size) <= 1.0:
            heads.append(dict(i=i, level=2, num=f"{ms.group(1)}.{ms.group(2)}{ms.group(3) or ''}",
                              title=ms.group(4).strip()))
    return heads


def bucket_of(title):
    t = title.lower()
    if t.startswith('abstract'):
        return 'Abstract'
    if 'introduction' in t:
        return 'Introduction'
    if re.search(r'related|prior work|previous work|literature', t):
        return 'Related Work'
    if re.search(r'conclu|limitation|future work|summary|broader impact|closing', t):
        return 'Conclusion'
    if re.search(r'experiment|implementation|evaluation|result|ablation|benchmark|comparison', t):
        return 'Experiments'
    if re.search(r'analys|discussion|stud(y|ies)|empirical|validation|application', t):
        return 'Experiments?'  # weak: Method if before the first Experiments section
    if re.search(r'acknowledg|llm usage|statement|ethic|reproducibility', t):
        return 'Ack'
    if re.search(r'method|approach|preliminar|background|formulation|framework|model|proposed|'
                 r'overview|our |architecture|pipeline|problem|setup|theory|analysis', t):
        return 'Method'
    return None  # decided by position


def analyse(pdf):
    pages, runs = read_runs(pdf)
    lines, body_stem, body_size = build_lines(pages, runs)
    classify_lines(lines, body_stem, body_size)
    heads = find_headings(lines, body_stem, body_size)
    res = dict(file=pdf, body_font=f'{body_stem}@{body_size}', n_pages=len(pages), fail=None)
    # column geometry from prose lines
    pl = [L for L in lines if L['prose']]
    for c in (0, 1):
        xs = Counter(round(L['x0']) for L in pl if L['col'] == c)
        res[f'colLeft{c}'] = xs.most_common(1)[0][0]
        xe = Counter(round(L['x1']) for L in pl if L['col'] == c)
        res[f'colRight{c}'] = xe.most_common(1)[0][0]
    colTop = min(L['top'] for L in pl if L['page'] > 1)
    colBot = max(L['bottom'] for L in pl)
    res['colTop'], res['colBot'] = colTop, colBot

    def pos(L):
        f = (L['top'] - colTop) / (colBot - colTop)
        return 2 * (L['page'] - 1) + L['col'] + min(max(f, 0.0), 1.0)

    # reading-order stream: columns 0/1 only (full-width items handled separately)
    order = [i for i, L in enumerate(lines) if L['col'] in (0, 1)]
    order.sort(key=lambda i: (lines[i]['page'], lines[i]['col'], lines[i]['top']))
    rank = {i: k for k, i in enumerate(order)}
    tops = sorted([h for h in heads if h['level'] == 1], key=lambda h: rank[h['i']])
    ref = [h for h in tops if re.match(r'(references|bibliography)', h['title'], re.I)]
    if not ref:
        res['fail'] = 'no References heading'
        return res, lines, heads
    ref = ref[0]
    body_tops = [h for h in tops if rank[h['i']] < rank[ref['i']]]
    if not body_tops or bucket_of(body_tops[0]['title']) != 'Abstract':
        res['fail'] = 'no Abstract heading first'
        return res, lines, heads
    nums = [h['num'] for h in body_tops if isinstance(h['num'], int)]
    if nums != list(range(1, len(nums) + 1)):
        res['fail'] = f'section numbers not 1..n: {nums}'
        return res, lines, heads
    # buckets: keyword, else by position (before first Experiments -> Method, after -> Experiments)
    bks = [bucket_of(h['title']) for h in body_tops]
    first_exp = next((k for k, b in enumerate(bks) if b == 'Experiments'), None)
    if first_exp is None:
        first_exp = next((k for k, b in enumerate(bks) if b == 'Experiments?'), None)
    for k, b in enumerate(bks):
        if b is None or b == 'Experiments?':
            bks[k] = 'Method' if first_exp is None or k < first_exp else 'Experiments'
    for h, b in zip(body_tops, bks):
        h['bucket'] = b
    res['sections'] = ' | '.join(f"{h['num'] if h['num'] is not None else '-'} {h['title']} -> {h['bucket']}"
                                 for h in body_tops)
    # caption blocks (excluded from prose even at body size)
    for k, i in enumerate(order):
        L = lines[i]
        L['caption'] = None
        if CAPTION_RE.match(L['text']):
            j = order[k - 1] if k else None
            P = lines[j] if j is not None else None
            cont = (P is not None and P['prose'] and L['prose'] and P['page'] == L['page'] and P['col'] == L['col']
                    and L['top'] - P['bottom'] < 0.6 * body_size)
            if not cont:
                L['caption'] = CAPTION_RE.match(L['text']).group(1)
    for k, i in enumerate(order):
        L = lines[i]
        if L['caption'] and L['prose']:
            # mark continuation lines of a body-size caption
            prev = L
            for j in order[k + 1:]:
                M = lines[j]
                if M['page'] != L['page'] or M['col'] != L['col'] or M['top'] - prev['bottom'] > 0.6 * body_size:
                    break
                M['prose'] = False
                prev = M
            L['prose'] = False
    # sections
    bounds = [rank[h['i']] for h in body_tops] + [rank[ref['i']]]
    secs = []
    for k, h in enumerate(body_tops):
        idx = order[bounds[k] + 1:bounds[k + 1]]
        secs.append(dict(h=h, lines=[lines[i] for i in idx],
                         span=pos(lines[order[bounds[k + 1]]]) - pos(lines[h['i']])))
    res['body_pages'] = pos(lines[ref['i']]) / 2
    last = [L for s in secs if s['h']['bucket'] in BUCKETS for L in s['lines'] if L['prose']][-1]
    f = (last['bottom'] - colTop) / (colBot - colTop)
    res['text_end_pages'] = (2 * (last['page'] - 1) + last['col'] + min(max(f, 0.0), 1.0)) / 2
    res['front_cols'] = pos(lines[body_tops[0]['i']])
    for b in BUCKETS + ['Ack']:
        res[f'words_{b}'] = 0
        res[f'lines_{b}'] = 0
        res[f'cols_{b}'] = 0.0
    for s in secs:
        b = s['h']['bucket']
        txt = '\n'.join(L['text'] for L in s['lines'] if L['prose'])
        s['words'] = count_words(txt)
        s['nlines'] = sum(L['prose'] for L in s['lines'])
        res[f'words_{b}'] += s['words']
        res[f'lines_{b}'] += s['nlines']
        res[f'cols_{b}'] += s['span']
    tot = sum(res[f'words_{b}'] for b in BUCKETS)
    res['words_body'] = tot
    res['lines_body'] = sum(res[f'lines_{b}'] for b in BUCKETS)
    for b in BUCKETS:
        res[f'share_{b}'] = res[f'words_{b}'] / tot if tot else float('nan')
    res['words_per_line'] = tot / res['lines_body'] if res['lines_body'] else float('nan')
    # sub-sections per bucket (level 2 headings inside the body)
    res['n_subsections_Method'] = res['n_subsections_Experiments'] = 0
    for k, h in enumerate(body_tops):
        n = sum(1 for g in heads if g['level'] == 2 and bounds[k] < rank[g['i']] < bounds[k + 1])
        if h['bucket'] in ('Method', 'Experiments'):
            res[f'n_subsections_{h["bucket"]}'] += n
    # Related Work position
    bl = [h['bucket'] for h in body_tops]
    res['mapping_ok'] = all(b in bl for b in ('Introduction', 'Method', 'Experiments'))
    if 'Related Work' not in bl:
        res['rw_position'] = 'none'
    else:
        k = bl.index('Related Work')
        if k > 0 and bl[k - 1] == 'Introduction':
            res['rw_position'] = 'after_intro'
        elif k + 1 < len(bl) and bl[k + 1] == 'Conclusion':
            res['rw_position'] = 'before_conclusion'
        else:
            res['rw_position'] = 'other'
    # figures / tables in the body (column lines before References + full-width lines on earlier pages
    # or above the References heading on the same page)
    R = lines[ref['i']]
    figs, tabs = set(), set()
    for L in lines:
        m = CAPTION_RE.match(L['text'])
        if not m:
            continue
        if L['col'] in (0, 1):
            if L.get('caption') is None or rank[lines.index(L)] > rank[ref['i']]:
                continue
        else:
            if L['page'] > R['page'] or (L['page'] == R['page'] and L['top'] > R['top'] and R['col'] == 0):
                continue
        (figs if m.group(1).startswith('F') else tabs).add(int(m.group(2)))
    # numbering is sequential, so the highest caption number is the count (a caption merged into a
    # neighbouring float's line can be missed; gaps are visible in fig_nums / tab_nums)
    res['n_figures'], res['n_tables'] = max(figs, default=0), max(tabs, default=0)
    res['fig_nums'] = ','.join(map(str, sorted(figs)))
    res['tab_nums'] = ','.join(map(str, sorted(tabs)))
    # Introduction paragraphs and bullets
    intro = [s for s in secs if s['h']['bucket'] == 'Introduction']
    if intro:
        p, b, paras = intro_structure(intro[0]['lines'], res, body_size)
        res['intro_paragraphs'], res['intro_bullets'] = p, b
        res['_intro_paras'] = paras
    # references
    after = [h for h in tops if rank[h['i']] > rank[ref['i']]]
    end = rank[after[0]['i']] if after else len(order)
    labs = []
    for i in order[rank[ref['i']] + 1:end]:
        m = re.match(r'^\[(\d+)\]', lines[i]['text'])
        if m:
            labs.append(int(m.group(1)))
    res['n_references'] = max(labs) if labs else None
    res['ref_labels_unique'] = len(set(labs))
    res['_secs'] = secs
    return res, lines, heads


def count_words(txt):
    txt = re.sub(r'(\w)-\n(\w)', r'\1\2', txt)
    return sum(1 for t in txt.split() if re.search(r'[A-Za-z0-9]', t))


def intro_structure(lines, res, body_size):
    """Paragraphs and list items of a section.

    A paragraph starts at the first prose line, at an indented line, or at a margin line that follows a
    short line (parskip style / after a list). Items are lines starting with a bullet, or with an
    enumeration label ("1.", "(1)", "i)") right after a short line or another item; item continuation
    lines share the item's hanging indent. A paragraph made only of a bold run-in heading (e.g.
    "Contributions." directly followed by a list) is not counted."""
    pl = [L for L in lines if L['prose']]
    paras, items = [], 0
    in_list, item_off, label_off, prev, prev_off = False, None, None, None, None
    for k, L in enumerate(pl):
        off = L['x0'] - res[f'colLeft{L["col"]}']
        prev_short = prev is not None and prev['x1'] < res[f'colRight{prev["col"]}'] - 20
        t = L['text']
        enum_item = ENUM_RE.match(t) and (prev_short or (in_list and abs(off - label_off) <= 1.5))
        if enum_item and k + 1 < len(pl) and L['x1'] >= res[f'colRight{L["col"]}'] - 20:
            # a numbered run-in paragraph ("1) Overload: ...") continues at the margin; an item hangs
            nxt = pl[k + 1]
            enum_item = nxt['x0'] - res[f'colLeft{nxt["col"]}'] > 3.0 or ENUM_RE.match(nxt['text'])
        if t.startswith(BULLETS) or enum_item:
            items += 1
            in_list, item_off, label_off, prev, prev_off = True, None, off, L, off
            continue
        left_list = False
        if in_list:
            # item continuation: hanging indent (relative to the column) after a full line; a line after
            # a short line ends the item, and a non-item line then starts a paragraph
            if item_off is None and off > 3 and not prev_short:
                item_off = off
            if item_off is not None and abs(off - item_off) <= 2.0 and not prev_short:
                prev, prev_off = L, off
                continue
            in_list, left_list = False, True
        indented = 3.0 < off < 30.0 and (left_list or prev_off is None or abs(off - prev_off) > 1.5)
        if prev is None or indented or (off <= 3.0 and prev_short) or not paras:
            paras.append([L])
        else:
            paras[-1].append(L)
        prev, prev_off = L, off
    real = [ps for ps in paras if not all(x['all_bold'] for x in ps)]
    return len(real), items, [' '.join(x['text'] for x in ps)[:70] for ps in real]


# ----------------------------------------------------------------------------------------------
def dump(pdf):
    res, lines, heads = analyse(pdf)
    print('#', pdf)
    print({k: v for k, v in res.items() if not k.startswith('_')})
    if res['fail']:
        for h in heads:
            print('  head', h['level'], h['num'], h['title'], lines[h['i']]['page'], lines[h['i']]['col'])
        return
    for s in res['_secs']:
        h = s['h']
        L = lines[h['i']]
        print(f"== {h['num']} {h['title']} [{h['bucket']}] p{L['page']} c{L['col']} y{L['top']:.0f} "
              f"words={s['words']} lines={s['nlines']} cols={s['span']:.2f}")
        for g in heads:
            if g['level'] == 2 and lines[g['i']] in s['lines']:
                print('     ', g['num'], g['title'])
    print('-- intro paragraphs:')
    for p in res.get('_intro_paras', []):
        print('   *', p)


def collect(paths, group):
    rows = []
    for pdf in paths:
        try:
            res, _, _ = analyse(pdf)
        except Exception as e:  # noqa: BLE001 - report and drop
            res = dict(file=pdf, fail=f'error: {e!r}')
        res['group'] = group
        res['paper'] = (os.path.basename(os.path.dirname(pdf)) if os.path.basename(pdf) == 'main.pdf'
                        else os.path.basename(pdf).replace('_CVPR_2025_paper.pdf', '').replace('.pdf', ''))
        rows.append(res)
        print(f"{group:8s} {res['paper'][:50]:50s} {res['fail'] or 'ok'}", file=sys.stderr)
    return rows


METRICS = (['body_pages', 'text_end_pages', 'words_body', 'lines_body', 'words_per_line']
           + [f'words_{b}' for b in BUCKETS] + [f'share_{b}' for b in BUCKETS]
           + [f'lines_{b}' for b in BUCKETS] + [f'cols_{b}' for b in BUCKETS]
           + ['n_figures', 'n_tables', 'intro_paragraphs', 'intro_bullets', 'n_references',
              'n_subsections_Method', 'n_subsections_Experiments'])


def usable(rows, m):
    """Per-bucket metrics only for papers whose sections map onto Intro/Method/Experiments."""
    bucket_metric = m.split('_')[0] in ('words', 'share', 'lines', 'cols') and m not in ('words_body', 'lines_body')
    return [r[m] for r in rows if r.get(m) is not None and (r['mapping_ok'] or not bucket_metric)]


def summarise(rows):
    out = {}
    for m in METRICS:
        v = np.array(usable(rows, m), float)
        if len(v) == 0:
            continue
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        out[m] = dict(n=int(len(v)), median=q50, q25=q25, q75=q75, min=v.min(), max=v.max(), mean=v.mean())
    for m in ('rw_position',):
        out[m] = dict(Counter(r[m] for r in rows))
    out['contribution_list_share'] = float(np.mean([r['intro_bullets'] > 0 for r in rows]))
    return out


def percentile(v, ref):
    ref = np.asarray(ref, float)
    return 100.0 * (np.sum(ref < v) + 0.5 * np.sum(ref == v)) / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--dump', nargs='*')
    args = ap.parse_args()
    if args.dump:
        for p in args.dump:
            dump(p)
        return
    local = sorted(glob.glob(LOCAL_GLOB))
    sample = sorted(glob.glob(os.path.join(SAMPLE_DIR, '*.pdf')))
    rows = collect(local, 'related') + collect(sample, 'sample') + collect([OURS], 'ours')
    os.makedirs(args.out, exist_ok=True)
    cols = ['group', 'paper', 'fail', 'mapping_ok', 'body_font', 'n_pages', 'sections', 'rw_position', 'fig_nums', 'tab_nums',
            'ref_labels_unique'] + METRICS + [f'words_Ack', 'front_cols']
    with open(os.path.join(args.out, 'per_paper.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    ok = [r for r in rows if not r['fail']]
    corpus = {g: [r for r in ok if r['group'] == g] for g in ('related', 'sample')}
    corpus['all'] = corpus['related'] + corpus['sample']
    ours = [r for r in ok if r['group'] == 'ours']
    stats = dict(n_used={g: len(v) for g, v in corpus.items()},
                 dropped=[dict(paper=r['paper'], group=r['group'], reason=r['fail']) for r in rows if r['fail']],
                 groups={g: summarise(v) for g, v in corpus.items()})
    if ours:
        o = ours[0]
        stats['ours'] = {m: o.get(m) for m in METRICS + ['rw_position', 'sections']}
        stats['ours_percentile'] = {g: {m: percentile(o[m], usable(corpus[g], m))
                                        for m in METRICS if o.get(m) is not None} for g in corpus}
    with open(os.path.join(args.out, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=1, default=float)
    print(json.dumps(stats['n_used']), len(stats['dropped']), 'dropped', file=sys.stderr)


if __name__ == '__main__':
    main()
