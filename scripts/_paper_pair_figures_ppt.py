#!/usr/bin/env python3
"""혼동 후보 점수 행렬과 최종 판정 비교를 편집 가능한 PowerPoint로 만든다."""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex

from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (XL_CHART_TYPE, XL_TICK_MARK, XL_AXIS_CROSSES,
                            XL_TICK_LABEL_POSITION, XL_DATA_LABEL_POSITION, XL_MARKER_STYLE)
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

import _pdv2_pair_visualization as exp
from _paper_figure4 import text_box, line, STYLES
from _paper_pair_diagnostics import DEST, OUT, COLOURS, NAMES, plt
from _paper_pair_matrix_alternatives import ADOPTED, ROOT, MATRIX_STYLE, figure
MATRIX_OUT = ROOT/ADOPTED
from _paper_baseline_figure_preview import COLORS as PRECISION_COLORS, STYLES as PRECISION_STYLES


def box(slide, name: str, x: float, y: float, w: float, h: float, fill: str):
    """편집 가능한 선호 결과 셀을 더한다."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = name
    shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.fill.background(); shape._element.spPr.append(OxmlElement('a:effectLst'))
    return shape


def precision_slide(prs: Presentation) -> None:
    """전체 평균 곡선과 확대 곡선·시드 편차 경계를 편집 가능한 차트로 만든다."""
    source = exp.read(DEST/'data/final_precision_curves.json')
    x = np.array(source['precision_grid']) * 100
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dash = {'--': MSO_LINE_DASH_STYLE.DASH, ':': MSO_LINE_DASH_STYLE.ROUND_DOT,
            '-.': MSO_LINE_DASH_STYLE.DASH_DOT}
    for panel, title in enumerate([f"(a) Final detection: NSD and {len(source['methods']) - 1} baselines", '(b) High-precision detail']):
        y = .07 + panel*2.06
        text_box(slide, f'precision_title_{panel}', .1, y, 3.15, .18, title, 8, bold=True)
        data = XyChartData()
        styles = []
        for method, row in source['methods'].items():
            series = data.add_series(row['label'])
            for p, recall in zip(x, row['mean']):
                series.add_data_point(float(p), float(recall*100))
            styles.append((method, False))
        if panel:
            for method in ('nsd', 'bevplace'):
                row = source['methods'][method]
                for sign in (-1, 1):
                    series = data.add_series(row['label'] + (' - SD' if sign < 0 else ' + SD'))
                    bound = (np.array(row['mean']) + sign*np.array(row['std']))*100
                    for p, recall in zip(x, bound):
                        series.add_data_point(float(p), float(recall))
                    styles.append((method, True))
        shape = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
                    Inches(.04), Inches(y+.22), Inches(3.27), Inches(1.76), data)
        shape.name = f'final_precision_{panel}'
        chart = shape.chart
        chart.has_legend = False
        chart.chart_style = 2
        for series, (method, is_bound) in zip(chart.series, styles):
            series.format.line.color.rgb = RGBColor.from_string(PRECISION_COLORS[method][1:])
            series.format.line.width = Pt(.35 if is_bound else 1.8 if method == 'nsd' else .9)
            style = ':' if is_bound else PRECISION_STYLES[method]
            if style in dash:
                series.format.line.dash_style = dash[style]
            series.smooth = False
        for axis, minimum, maximum, step, label in (
            (chart.category_axis, 95 if panel else 0, 100, 1 if panel else 25, 'Required precision (%)'),
            (chart.value_axis, 53 if panel else 0, 85 if panel else 100, 10 if panel else 25, 'Recall (%)'),
        ):
            axis.minimum_scale = minimum
            axis.maximum_scale = maximum
            axis.major_unit = step
            axis.tick_labels.font.name = 'Calibri'
            axis.tick_labels.font.size = Pt(6)
            axis.tick_labels.number_format = '0'
            axis.major_tick_mark = XL_TICK_MARK.OUTSIDE
            axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
            axis.crosses = XL_AXIS_CROSSES.MINIMUM
            axis.format.line.color.rgb = RGBColor.from_string('6D7780')
            axis.format.line.width = Pt(.6)
            axis.has_major_gridlines = True
            axis.major_gridlines.format.line.color.rgb = RGBColor.from_string('E8E8E8')
            axis.major_gridlines.format.line.width = Pt(.5)
            axis.has_title = True
            axis.axis_title.text_frame.text = label
            for paragraph in axis.axis_title.text_frame.paragraphs:
                paragraph.font.name = 'Calibri'
                paragraph.font.size = Pt(7)
        if panel:
            for method, left, top in (('nsd', 2.02, 3.12), ('bevplace', 1.82, 3.54)):
                row = source['methods'][method]
                text_box(slide, 'endpoint_'+method, left, top, 1.32, .15,
                         f"{row['label']} {row['mean'][-1]*100:.1f}%", 6.5,
                         bold=True, color=PRECISION_COLORS[method][1:])
    order = ['nsd', 'bevplace'] + [m for m in source['methods'] if m not in ('nsd', 'bevplace')]
    for i, method in enumerate(order):
        x0, y = .12+(i % 3)*1.09, 4.24+(i//3)*.14
        mark = line(slide, 'legend_'+method, x0, y, x0+.17, y, PRECISION_COLORS[method][1:],
                    1.8 if method == 'nsd' else .9)
        mark._element.spPr.append(OxmlElement('a:effectLst'))
        if PRECISION_STYLES[method] in dash:
            mark.line.dash_style = dash[PRECISION_STYLES[method]]
        text_box(slide, 'legend_label_'+method, x0+.21, y-.07, .85, .14,
                 source['methods'][method]['label'], 6.2, bold=method == 'nsd')
    text_box(slide, 'precision_scope', .1, 4.76, 3.15, .12,
             'Thin bounds in (b): ±1 seed SD (shaded in the PDF).', 5.8)


def manual_plot_area(chart, x: float, y: float, w: float, h: float) -> None:
    """차트 안 축 사각형(inner plot area)을 차트 틀 기준 비율로 고정한다."""
    from pptx.oxml.ns import qn
    plot = chart._chartSpace.chart.plotArea
    old = plot.find(qn('c:layout'))
    if old is not None:
        plot.remove(old)
    layout, manual = OxmlElement('c:layout'), OxmlElement('c:manualLayout')
    for tag, value in (('layoutTarget', 'inner'), ('xMode', 'edge'), ('yMode', 'edge'),
                       ('x', x), ('y', y), ('w', w), ('h', h)):
        child = OxmlElement('c:' + tag)
        child.set('val', value if isinstance(value, str) else f'{value:.6f}')
        manual.append(child)
    layout.append(manual)
    plot.insert(0, layout)


# PowerPoint(Microsoft 365, 이 PC)는 수동 축 사각형을 차트 틀 기준 (−0.042, +0.022)인치 옮겨 그린다.
# 내보낸 PDF에서 잰 값이며, 차트 틀을 반대로 옮겨 축 사각형을 PDF 그림 좌표에 맞춘다.
PPT_PLOT_SHIFT = (0., 0.)


def no_fill(chart) -> None:
    """차트 영역 배경과 테두리를 비워 겹친 틀이 다른 요소를 가리지 않게 한다."""
    from pptx.oxml.ns import qn
    space = chart._chartSpace
    old = space.find(qn('c:spPr'))
    if old is not None:
        space.remove(old)
    sp = OxmlElement('c:spPr')
    sp.append(OxmlElement('a:noFill'))
    ln = OxmlElement('a:ln')
    ln.append(OxmlElement('a:noFill'))
    sp.append(ln)
    # c:spPr는 c:chart 뒤, c:txPr·c:externalData 앞에 둔다.
    space.insert(list(space).index(space.find(qn('c:chart'))) + 1, sp)


def precision_geometry(layout: dict, source: dict, fonts: dict) -> dict:
    """PDF 그림 5와 같은 matplotlib 배치를 데이터 없이 그려 제목·축 제목·끝점 라벨·범례 글자 위치(인치)를 얻는다."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.lines import Line2D
    from _paper_baseline_figure_review import LABELS
    plt.rcParams.update({"font.family": "Carlito", "font.size": fonts["label"], "text.color": "#0E2841",
                         "axes.labelcolor": "#0E2841", "axes.edgecolor": "#6D7780", "axes.linewidth": .6})
    fig = plt.figure(figsize=layout['figsize'])
    fig_h = layout['figsize'][1]
    overview, detail = fig.add_axes(layout['overview']), fig.add_axes(layout['detail'])
    (x0, x1), (y0, y1) = layout['zoom_x'], layout['zoom_y']
    overview.set(xlim=(0, 100), ylim=(0, 100), xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50, 75, 100])
    detail.set(xlim=(x0, x1), ylim=(y0, y1), xticks=layout['zoom_xticks'], yticks=layout['zoom_yticks'])
    notes = {}
    for method in ('nsd', 'bevplace'):
        value = source['methods'][method]['mean'][-1]*100
        notes[method] = detail.annotate(f"{LABELS[method]} {value:.1f}%", (100, value), xytext=layout['endpoint_offsets'][method],
                                        textcoords="offset points", ha="right", fontsize=fonts["endpoint"], weight="bold")
    titles = (f"(a) Final detection: NSD and {len(source['methods']) - 1} baselines", "(b) High-precision detail")
    for ax, title in zip((overview, detail), titles):
        ax.set(xlabel="Required precision (%)", ylabel="Recall (%)")
        ax.set_title(title, fontsize=fonts["title"], weight="bold", pad=fonts["title_pad"])
        ax.tick_params(labelsize=fonts["tick"], length=2, pad=2)
    order = ["nsd", "bevplace"] + [m for m in LABELS if m in source["methods"] and m not in ("nsd", "bevplace")]
    handles = [Line2D([0], [0], label=LABELS[m]) for m in order]
    legend = fig.legend(handles=handles, frameon=False, fontsize=fonts["legend"], handlelength=layout.get("handlelength", 2),
                        handletextpad=.5, labelspacing=.35, **layout["legend"])
    legend.get_texts()[0].set_weight("bold")
    fig.canvas.draw()
    renderer, inverse = fig.canvas.get_renderer(), fig.dpi_scale_trans.inverted()

    def place(artist) -> tuple:
        b = artist.get_window_extent(renderer).transformed(inverse)
        return b.x0, fig_h - b.y1, b.width, b.height
    geometry = dict(order=order, titles=[(place(ax.title), ax.title.get_text()) for ax in (overview, detail)],
                    xlabels=[place(ax.xaxis.label) for ax in (overview, detail)],
                    ylabels=[place(ax.yaxis.label) for ax in (overview, detail)],
                    endpoints={m: place(a) for m, a in notes.items()},
                    legend=[place(t) for t in legend.get_texts()])
    plt.close(fig)
    return geometry


def precision_slide_layout(prs: Presentation, layout: dict) -> None:
    """PDF 그림 5의 배치(PRECISION_LAYOUT)대로 두 축·확대 상자·끝점·4열 범례를 편집 가능한 차트로 만든다.

    축 사각형은 PDF와 같은 그림 좌표(인치)에 고정하고, 제목·축 제목·끝점 라벨·범례 글자는 같은 배치의
    matplotlib 위치에 둔다. PowerPoint 차트는 띠 채우기를 지원하지 않아 시드 편차는 얇은 경계선으로 둔다.
    """
    source = exp.read(DEST/'data/final_precision_curves.json')
    x = np.array(source['precision_grid']) * 100
    colors = {**PRECISION_COLORS, **layout.get('colors', {})}
    fonts = {**dict(tick=6.5, legend=6.5, label=7, title=7.7, endpoint=7, title_pad=6), **layout.get('fonts', {})}
    lw_base = layout.get('lw_base', .9)
    width = {m: 1.8 if m == 'nsd' else 1.35 if m == 'bevplace' else lw_base for m in source['methods']}
    fig_w, fig_h = layout['figsize']
    geometry = precision_geometry(layout, source, fonts)
    ink = '0E2841'
    dash = {'--': MSO_LINE_DASH_STYLE.DASH, ':': MSO_LINE_DASH_STYLE.ROUND_DOT,
            '-.': MSO_LINE_DASH_STYLE.DASH_DOT}
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    def put(name: str, box: tuple, text: str, size: float, rotated: bool = False, **kwargs) -> None:
        bx, by, bw, bh = box
        long, short = (bh, bw) if rotated else (bw, bh)
        cx, cy = bx + bw/2, by + bh/2
        text_box(slide, name, cx - long/2 - .03, cy - short/2 - .01, long + .06, short + .02, text, size,
                 rotation=270 if rotated else 0, **kwargs)

    (x0, x1), (y0, y1) = layout['zoom_x'], layout['zoom_y']
    xt, yt = layout['zoom_xticks'], layout['zoom_yticks']
    panels = ((layout['overview'], (0, 100), (0, 100), 25, 25), (layout['detail'], (x0, x1), (y0, y1), xt[1] - xt[0], yt[1] - yt[0]))
    # 그리는 순서는 PDF의 겹침 순서(비교 방법 → BEVPlace++ → NSD)를 따른다.
    drawn = [m for m in source['methods'] if m not in ('nsd', 'bevplace')] + ['bevplace', 'nsd']
    boxes = []
    for panel, (rect, xlim, ylim, xstep, ystep) in enumerate(panels):
        # PDF 축 사각형(그림 비율)을 슬라이드 인치로 바꾼다. 차트 틀은 눈금 글자 자리를 둘러 더 크게 잡는다.
        left, top = rect[0]*fig_w, (1 - rect[1] - rect[3])*fig_h
        w, h = rect[2]*fig_w, rect[3]*fig_h
        # 틀 여백이 좁으면 PowerPoint가 눈금 글자 자리를 위해 축 사각형을 줄이므로 여백을 넉넉히 둔다.
        frame = (left - .45 - PPT_PLOT_SHIFT[0], top - .12 - PPT_PLOT_SHIFT[1], w + .6, h + .42)
        boxes.append((left, top, w, h, xlim, ylim))
        (title_box, title), xlabel, ylabel = geometry['titles'][panel], geometry['xlabels'][panel], geometry['ylabels'][panel]
        put(f'precision_title_{panel}', title_box, title, fonts['title'], bold=True, color=ink)
        put(f'precision_xlabel_{panel}', xlabel, 'Required precision (%)', fonts['label'], color=ink)
        # PowerPoint는 세로 눈금 글자를 matplotlib보다 약 0.04인치 더 왼쪽에 두므로 세로축 제목도 그만큼 옮긴다.
        put(f'precision_ylabel_{panel}', (ylabel[0] - .04, *ylabel[1:]), 'Recall (%)', fonts['label'], rotated=True, color=ink)
        data = XyChartData()
        styles = []
        if panel:
            for method in ('bevplace', 'nsd'):
                row = source['methods'][method]
                for sign in (-1, 1):
                    series = data.add_series(row['label'] + (' - SD' if sign < 0 else ' + SD'))
                    bound = (np.array(row['mean']) + sign*np.array(row['std']))*100
                    for p, recall in zip(x, bound):
                        series.add_data_point(float(p), float(recall))
                    styles.append((method, True))
        for method in drawn:
            row = source['methods'][method]
            series = data.add_series(row['label'])
            for p, recall in zip(x, row['mean']):
                series.add_data_point(float(p), float(recall*100))
            styles.append((method, False))
        shape = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
                                       *(Inches(v) for v in frame), data)
        shape.name = f'final_precision_{panel}'
        chart = shape.chart
        chart.has_legend = False
        chart.has_title = False
        chart.chart_style = 2
        no_fill(chart)
        manual_plot_area(chart, (left - frame[0])/frame[2], (top - frame[1])/frame[3], w/frame[2], h/frame[3])
        for series, (method, is_bound) in zip(chart.series, styles):
            series.format.line.color.rgb = RGBColor.from_string(colors[method][1:])
            series.format.line.width = Pt(.35 if is_bound else width[method])
            style = ':' if is_bound else PRECISION_STYLES[method]
            if style in dash:
                series.format.line.dash_style = dash[style]
            series.smooth = False
        for axis, (minimum, maximum), step in ((chart.category_axis, xlim, xstep), (chart.value_axis, ylim, ystep)):
            # PowerPoint 눈금은 축 최솟값에서 시작한다. 최솟값이 PDF 눈금에 없으면 절반 간격으로 둔다.
            if minimum % step:
                step = step / 2
            axis.minimum_scale, axis.maximum_scale, axis.major_unit = minimum, maximum, step
            axis.tick_labels.font.name = 'Calibri'
            axis.tick_labels.font.size = Pt(fonts['tick'])
            axis.tick_labels.number_format = '0'
            axis.tick_labels.number_format_is_linked = False
            axis.major_tick_mark = XL_TICK_MARK.OUTSIDE
            axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
            axis.crosses = XL_AXIS_CROSSES.MINIMUM
            axis.format.line.color.rgb = RGBColor.from_string('6D7780')
            axis.format.line.width = Pt(.6)
            axis.has_major_gridlines = True
            axis.major_gridlines.format.line.color.rgb = RGBColor.from_string('ECEFF1')
            axis.major_gridlines.format.line.width = Pt(.45)
            axis.has_title = False

    def to_slide(panel: int, px: float, py: float) -> tuple:
        left, top, w, h, (a, b), (c, d) = boxes[panel]
        return left + (px - a)/(b - a)*w, top + (d - py)/(d - c)*h

    # (a)의 확대 범위 상자
    zx0, zy0 = to_slide(0, x0, y1)
    zx1, zy1 = to_slide(0, x1, y0)
    zoom = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(zx0), Inches(zy0), Inches(zx1 - zx0), Inches(zy1 - zy0))
    zoom.name = 'zoom_box'
    zoom.fill.background()
    zoom.line.color.rgb = RGBColor.from_string('3B4650')
    zoom.line.width = Pt(.8)
    zoom._element.spPr.append(OxmlElement('a:effectLst'))
    # (b)의 끝점 점·지시선·라벨. 지시선은 PDF처럼 글자 상자의 relpos 지점에서 끝점까지 긋는다.
    for method in ('nsd', 'bevplace'):
        row = source['methods'][method]
        value = row['mean'][-1]*100
        px, py = to_slide(1, 100, value)
        bx, by, bw, bh = geometry['endpoints'][method]
        rx, ry = layout['endpoint_relpos'][method]
        put('endpoint_'+method, (bx, by, bw, bh), f"{row['label']} {value:.1f}%", fonts['endpoint'],
            bold=True, color=colors[method][1:])
        lead = line(slide, 'endpoint_leader_'+method, bx + rx*bw, by + (1 - ry)*bh, px, py, colors[method][1:], .6)
        lead._element.spPr.append(OxmlElement('a:effectLst'))
        size = 3.5/72
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px - size/2), Inches(py - size/2), Inches(size), Inches(size))
        dot.name = 'endpoint_dot_'+method
        dot.fill.solid(); dot.fill.fore_color.rgb = RGBColor.from_string(colors[method][1:])
        dot.line.color.rgb = RGBColor.from_string('FFFFFF'); dot.line.width = Pt(.4)
        dot._element.spPr.append(OxmlElement('a:effectLst'))
    # 4열 범례: matplotlib 범례 글자 위치에 두고, 선 표본은 글자 왼쪽(handletextpad 0.5, handlelength)에 둔다.
    em = fonts['legend']/72
    handle = layout.get('handlelength', 2)*em
    for method, (bx, by, bw, bh) in zip(geometry['order'], geometry['legend']):
        hx1, hy = bx - .5*em, by + bh/2
        mark = line(slide, 'legend_'+method, hx1 - handle, hy, hx1, hy, colors[method][1:],
                    1.8 if method == 'nsd' else max(1, lw_base))
        mark._element.spPr.append(OxmlElement('a:effectLst'))
        if PRECISION_STYLES[method] in dash:
            mark.line.dash_style = dash[PRECISION_STYLES[method]]
        text_box(slide, 'legend_label_'+method, bx - .01, by - .01, bw + .06, bh + .02, source['methods'][method]['label'],
                 fonts['legend'], bold=method == 'nsd', color=colors['nsd'][1:] if method == 'nsd' else ink, align=PP_ALIGN.LEFT)
    bottom = max(by + bh for _, by, _, bh in geometry['legend'])
    text_box(slide, 'precision_scope', .1, bottom + .08, fig_w - .2, .12,
             'Thin bounds in (b): ±1 seed SD (shaded in the PDF).', 5.8, color=ink)


def matrix_slide(prs: Presentation, style: dict | None = None) -> None:
    """PDF 그림 4의 축·글자 위치를 같은 크기로 옮겨 썸네일·행렬 셀·색 막대를 만든다.

    style을 주면 PDF와 같은 MATRIX_LAYOUT으로 배치·× 색과 크기·정답 테두리 외곽선을 맞춘다. 없으면 기존 배치다.
    """
    s = {**MATRIX_STYLE, **(style or {})}
    fig = figure(ADOPTED, style)
    fig.canvas.draw()
    renderer, inverse = fig.canvas.get_renderer(), fig.dpi_scale_trans.inverted()
    bound = fig.get_tightbbox(renderer)
    ox, top = (prs.slide_width.inches-bound.width)/2-bound.x0, .05+bound.y1
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    def place(artist) -> tuple:
        b = artist.get_window_extent(renderer).transformed(inverse)
        return ox+b.x0, top-b.y1, b.width, b.height

    def label(name: str, text) -> None:
        x, y, w, h = place(text)
        turned = bool(text.get_rotation())
        long, short = (h, w) if turned else (w, h)
        cx, cy = x+w/2, y+h/2
        text_box(slide, name, cx-long/2-.03, cy-short/2-.01, long+.06, short+.02, text.get_text(),
                 text.get_fontsize(), color=to_hex(text.get_color())[1:].upper(),
                 bold=text.get_weight() == 'bold', rotation=270 if turned else 0)

    for k, ax in enumerate(fig.axes[:10]):
        x, y, w, _ = place(ax)
        slide.shapes.add_picture(str(DEST/f'query_{k+1:02d}.png'), Inches(x), Inches(y), width=Inches(w))
        label(f'query_number_{k+1}', ax.title)
    matrix = exp.read(MATRIX_OUT/'matrix_results.json')
    same = np.array(matrix['selection']['gt_distance']) <= 5
    colour = colormaps['viridis']
    for method, ax in zip(exp.METHODS, fig.axes[10:16]):
        x, y, w, _ = place(ax)
        cell = w/10
        entry = matrix['methods'][method]
        ranks, native = np.array(entry['row_rank']), np.array(entry['native_score'])
        label('method_'+method, ax.title)
        for i, text in enumerate(ax.get_yticklabels()):
            label(f'{method}_row_{i}', text)
        for i, text in enumerate(ax.get_xticklabels()):
            label(f'{method}_col_{i}', text)
        # × 크기는 기존 비율(scatter 10 ↔ 6.6pt)을 유지해 scatter 면적의 제곱근으로 맞춘다.
        cross_pt = round(6.6*(s['cross_size']/MATRIX_STYLE['cross_size'])**.5, 1)

        def winner(i: int) -> None:
            text_box(slide, f'{method}_winner_{i}', x+int(np.argmax(native[i]))*cell, y+i*cell,
                     cell, cell, '×', cross_pt, color=s['cross_color'][1:].upper(), bold=True)
        for i in range(10):
            for j in range(10):
                rgb = ''.join(f'{round(255*c):02X}' for c in colour(ranks[i,j])[:3])
                shape = box(slide, f'matrix_{method}_{i}_{j}', x+j*cell, y+i*cell, cell, cell, rgb)
                if same[i,j] and not s['outline_halo']:
                    shape.line.color.rgb = RGBColor.from_string('FFFFFF'); shape.line.width = Pt(.6)
            if not s['outline_halo']:
                winner(i)
        if s['outline_halo']:
            # PDF처럼 정답 테두리는 모든 셀 위에 짙은 외곽선(halo)과 흰 선을 겹쳐 그린다.
            for i, j in zip(*np.nonzero(same)):
                for suffix, colour_hex, lw in (('halo', s['outline_halo'][1][1:], s['outline_halo'][0]), ('', 'FFFFFF', .65)):
                    frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x+j*cell), Inches(y+i*cell), Inches(cell), Inches(cell))
                    frame.name = f'matrix_{method}_{i}_{j}_outline' + ('_' + suffix if suffix else '')
                    frame.fill.background()
                    frame.line.color.rgb = RGBColor.from_string(colour_hex.upper()); frame.line.width = Pt(lw)
                    frame._element.spPr.append(OxmlElement('a:effectLst'))
            for i in range(10):
                winner(i)
    names = (('query_label',) if s['query_label'] else ()) + ('query_axis', 'reference_axis', 'matrix_legend')
    assert len(names) == len(fig.texts)
    for name, text in zip(names, fig.texts):
        label(name, text)
    cax = fig.axes[16]
    x, y, w, h = place(cax)
    for k in range(20):
        rgb = ''.join(f'{round(255*c):02X}' for c in colour(k/19)[:3])
        box(slide, f'colourbar_{k}', x+w*k/20, y, w/20, h, rgb)
    frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    frame.name = 'colourbar_outline'; frame.fill.background()
    frame.line.color.rgb = RGBColor.from_string('000000'); frame.line.width = Pt(.8)
    frame._element.spPr.append(OxmlElement('a:effectLst'))
    for k, tick in enumerate(cax.get_xticks()):
        mark = line(slide, f'rank_tickmark_{k}', x+w*tick, y+h, x+w*tick, y+h+2/72, '000000', .8)
        mark._element.spPr.append(OxmlElement('a:effectLst'))
    for k, text in enumerate(cax.get_xticklabels()):
        label(f'rank_tick_{k}', text)
    label('rank_label', cax.xaxis.label)
    plt.close(fig)


def main() -> None:
    """여섯 행렬의 글자·셀과 두 최종 판정 차트의 원좌표를 PPTX에 포함한다."""
    prs = Presentation(); prs.slide_width = Inches(3.35); prs.slide_height = Inches(4.9)
    matrix_slide(prs)
    precision_slide(prs)
    path = DEST/'pair_figures_editable.pptx'; prs.save(path)
    methods = len(exp.read(DEST/'data/final_precision_curves.json')['methods'])
    exp.write(DEST/'ppt_manifest.json', dict(file=str(path.relative_to(exp.REPO)), sha256=exp.sha(path),
              slides=2, matrix_cells=600, native_xy_charts=2, native_xy_series=2*methods+4, mean_series=2*methods, sd_boundary_series=4, lidar_thumbnails=10,
              precision_axis_unit='percent', precision_axis_ranges=[[0, 100], [95, 100]], direct_endpoint_labels=2,
              slide_size_inches=[3.35, 4.9],
              input_sources=[str((MATRIX_OUT/'matrix_results.json').relative_to(exp.REPO)), 'final_precision_curves.json'],
              note='글자·행렬 셀·최종 판정 곡선은 편집 가능하다. 첫 장은 PDF 그림 4와 같은 위치·글자 크기(썸네일 한 줄, 행렬 3열×2행)이며 '
                   '아래쪽은 비어 있다. 점군은 실제 입력의 래스터 표시다. 확대 차트의 시드 편차는 PDF의 띠와 같은 경계를 얇은 선으로 표시한다.'))
    print(path)


if __name__ == '__main__':
    main()
