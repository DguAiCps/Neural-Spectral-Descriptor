"""fig_intro_min2.pptx를 LibreOffice Impress로 PDF 내보낼 때 깨지는 부분을 고친 사본을 만든다(원본은 그대로).
1) 수식 개체(a14:m) 8개: LibreOffice는 대체 PNG(56x53 px 등)로 그리거나(□) 글꼴 대체에 실패한다.
   -> 일반 글자 런으로 바꾼다(수학 영숫자 기호는 일반 글자 + 굵게/기울임, Cambria; 아래첨자는 baseline; 물결은 결합 문자 U+0303).
2) 글상자 3개가 LibreOffice 글자 폭 차이로 줄바꿈된다 -> 폭을 25% 넓힌다(가운데 정렬은 중심 유지).
사용: python3 scripts/_fig_intro_libreoffice_fix_20261117.py docs/paper/body_v2.7/figs/fig_intro_min2.pptx docs/paper/body_v2.7/figs/fig_intro_min2_lo.pptx
"""
import re, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src); s = z.read('ppt/slides/slide1.xml').decode()
MAP = {'\U0001D485': 'd', '\U0001D434': 'A', '\U0001D435': 'B', '\U0001D49B': 'z'}
NOWRAP = ('invariant-key space', 'refined-spectrum space', 'b) NSD: phase selects, context accepts')

def math_to_runs(m):
    out, sub, acc = [], 0, 0
    for tok in re.finditer(r'<m:sub>|</m:sub>|<m:acc>|</m:acc>|<m:r>(<a:rPr[^>]*?(?:/>|>.*?</a:rPr>))<m:t>(.*?)</m:t></m:r>', m.group(0), re.S):
        g = tok.group(0)
        if g in ('<m:sub>', '</m:sub>'): sub += 1 if g == '<m:sub>' else -1; continue
        if g in ('<m:acc>', '</m:acc>'): acc += 1 if g == '<m:acc>' else -1; continue
        rpr, t = tok.group(1).replace(' smtClean="0"', ''), tok.group(2)
        if not t.strip(' \xa0'):
            out.append(('space', rpr, t)); continue
        for a, b in MAP.items(): t = t.replace(a, b)
        assert t.isascii(), repr(t)
        if acc: t += '̃'
        rpr = rpr.replace('typeface="Cambria Math"', 'typeface="Cambria"')
        if ' i="' not in rpr: rpr = rpr.replace('<a:rPr ', '<a:rPr i="1" ', 1)
        if sub:
            rpr = re.sub(r' baseline="[^"]*"', '', rpr, count=1).replace('<a:rPr ', '<a:rPr baseline="-25000" ', 1)
        out.append(('text', rpr, t))
    return ''.join(f'<a:r>{r}<a:t>{t}</a:t></a:r>' for _, r, t in out)

def fix_sp(sp):
    b = sp.group(0)
    if '<a14:m' in b:
        b = re.sub(r'<a14:m[ >].*?</a14:m>', math_to_runs, b, flags=re.S)
        if 'prst="ellipse"' in b:  # 원 안: 앞쪽 공백 런을 버리고 줄바꿈을 끈다
            b = re.sub(r'<a:r><a:rPr[^>]*>(?:(?!</a:r>).)*?<a:t>[ \xa0]+</a:t></a:r>', '', b, flags=re.S)
            b = re.sub(r'<a:bodyPr(?![^>]*wrap=)', '<a:bodyPr wrap="none"', b, count=1)
    if any(f'<a:t>{t}</a:t>' in b for t in NOWRAP):  # 폭 25% 넓힘(가운데 정렬이면 중심 유지)
        x, y = map(int, re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"/>', b).groups())
        cx, cy = map(int, re.search(r'<a:ext cx="(\d+)" cy="(\d+)"/>', b).groups())
        extra = cx // 4
        nx = x - extra // 2 if 'algn="ctr"' in b else x
        b = b.replace(f'<a:off x="{x}" y="{y}"/>', f'<a:off x="{nx}" y="{y}"/>', 1).replace(f'<a:ext cx="{cx}" cy="{cy}"/>', f'<a:ext cx="{cx + extra}" cy="{cy}"/>', 1)
    return b

n_alt = len(re.findall(r'<mc:AlternateContent', s))
s = re.sub(r'<mc:AlternateContent.*?</mc:AlternateContent>',
           lambda b: re.search(r'<mc:Choice[^>]*>(.*?)</mc:Choice>', b.group(0), re.S).group(1), s, flags=re.S)
n_math = len(re.findall(r'<a14:m[ >]', s))
s = re.sub(r'<p:sp>.*?</p:sp>', fix_sp, s, flags=re.S)
assert not re.search(r'<a14:m[ >]', s) and n_alt == 4 and n_math == 8, (n_alt, n_math)
import xml.dom.minidom; xml.dom.minidom.parseString(s.encode())  # XML 검사
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
    for i in z.infolist():
        o.writestr(i, s.encode() if i.filename == 'ppt/slides/slide1.xml' else z.read(i.filename))
print('alternate', n_alt, 'math', n_math, 'nowrap', sum(s.count(f'<a:t>{t}</a:t>') for t in NOWRAP))
