#!/usr/bin/env python3
"""'BEVPlace++ + 같은 시간 판정' 대조에 R@99P·R@95P를 더한다(원고 제외 대조, 반박 대비용).

`_pdv2_bev_temporal_current.py`와 같은 원장·질의·집계(시퀀스 → 6도시 → 시작점 3개)와 같은 구간 절차
(도시 안 양성 시퀀스 복원추출 10,000회, 시드 20261104)를 쓴다. 새 학습·추론은 없다.
검산: AP·최대 F1·R@100P 평균이 기존 summary.json과 일치해야 한다.
"""
import json
from pathlib import Path

import numpy as np

from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import classify_queries, pr_sweep

ROOT = Path(__file__).resolve().parents[1]
NSD = ROOT / "results/pdv2_diffattn_contrast_test_20261104/test"
BEV = ROOT / "results/pdv2_review_response_20261104/bev/test"
OUT = ROOT / "results/pdv2_bev_temporal_current_20261106"
METRICS = ("AP", "max_F1", "R@95P", "R@99P", "R@100P", "R@1")
TABLE = ROOT / "docs/paper/body_v2.3/tables_v3/appdx_bev_temporal.tex"
SEEDS = (1, 2, 3)


def table(out: dict) -> None:
    """부록 표(평균과 짝지은 차이)를 쓴다."""
    def signed(x: float) -> str:
        return ("$+$" if round(x, 2) >= 0 else "$-$") + f"{abs(x):.2f}"
    rows = [name + " & " + " & ".join(f"{100 * out['mean'][c][m]:.2f}" for m in METRICS) + r" \\"
            for name, c in (("BEVPlace++", "bev"), ("BEVPlace++ + TE", "bevt"), ("NSD", "diff"))]
    rows.append(r"\midrule")
    for name, key in (("(BEVPlace++ + TE) minus BEVPlace++", "bevt-bev"), ("NSD minus (BEVPlace++ + TE)", "diff-bevt")):
        v = out["comparisons"][key]
        rows.append(r"\raggedright " + name + " & " + " & ".join(
            f"{signed(v['delta_pp'][m])} [{signed(v['conditional_95_pp'][m][0])}, {signed(v['conditional_95_pp'][m][1])}]" for m in METRICS) + r" \\")
    TABLE.write_text(r"""% scripts/_pdv2_bev_temporal_highprec.py가 results/pdv2_bev_temporal_highprec_20261115/highprec.json에서 생성한다. 숫자를 직접 고치지 않는다.
\begin{table*}[t]
\caption{BEVPlace++ with distance-matched temporal evidence (TE) (\%); the lower rows are paired differences with conditional 95\% intervals (percentage points).}\label{tab:bev_temporal}
\centering\footnotesize\setlength{\tabcolsep}{2pt}
\sbox0{%
\begin{tabular}{@{}p{130pt}cccccc@{}}\toprule
Method or comparison & AP & Max F1 & R@95P & R@99P & R@100P & Revisit R@1 \\
\midrule
@ROWS@
\bottomrule\end{tabular}}
\ifdim\wd0>\textwidth\resizebox{\textwidth}{!}{\usebox0}\else\usebox0\fi
\end{table*}
""".replace("@ROWS@", "\n".join(rows)))


def main() -> None:
    per = {(c, s): {} for c in ("diff", "bevt", "bev") for s in SEEDS}
    count = 0
    for sensor, seq in TEST:
        name = f"{sensor}_{seq}"
        for s in SEEDS:
            ref = np.load(NSD / name / f"diff_s{s}.npz"); led = np.load(BEV / f"s{s}" / f"{name}.npz")
            for f in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(led[f], ref[f])
            qc = classify_queries(ref["min_past_gt_dist"])
            if not qc["n_pos"]:
                continue
            for c, d, key in (("diff", ref, "score"), ("bevt", led, "score"), ("bev", led, "score_negl2")):
                sw = pr_sweep(d[key], d["top1_geo_dist"], qc)
                per[(c, s)][f"{sensor}/{seq}"] = [sw[m] for m in METRICS[:-1]] + [float((d["top1_geo_dist"][qc["positive"]] <= 5).mean())]
        count += len(ref["query_idx"])
    assert count == 195726
    cities = {c: [s for s in names if s in per[("diff", 1)]] for c, names in CITY.items()}
    mean = {c: np.mean([[np.mean([per[(c, s)][q] for q in v], 0) for v in cities.values()] for s in SEEDS], (0, 1))
            for c in ("diff", "bevt", "bev")}
    old = json.loads((OUT / "summary.json").read_text())["mean"]
    for c in mean:
        for m in ("AP", "max_F1", "R@100P", "R@1"):
            assert abs(mean[c][METRICS.index(m)] - old[c][m]) < 1e-12, (c, m)
    out = dict(queries_per_case=count, mean={c: dict(zip(METRICS, v.tolist())) for c, v in mean.items()}, comparisons={},
               intervals=dict(replicates=10000, seed=20261104, method="도시 안 양성 시퀀스 복원추출, 세 시작점 평균 차이, 도시 동일 비중"),
               scope="반박 대비용. 원고 제외 대조이며 시험 선택에 쓰지 않는다.")
    for a, b in (("diff", "bev"), ("diff", "bevt"), ("bevt", "bev")):
        rng = np.random.default_rng(20261104); draws = []
        for v in cities.values():
            mat = np.array([np.mean([100 * (np.array(per[(a, s)][q]) - per[(b, s)][q]) for s in SEEDS], 0) for q in v])
            draws.append(mat[rng.integers(0, len(v), size=(10000, len(v)))].mean(1))
        boot = np.stack(draws).mean(0)
        out["comparisons"][f"{a}-{b}"] = dict(delta_pp=dict(zip(METRICS, (100 * (mean[a] - mean[b])).tolist())),
                                              conditional_95_pp={m: np.quantile(boot[:, i], [.025, .975]).tolist() for i, m in enumerate(METRICS)})
    (ROOT / "results/pdv2_bev_temporal_highprec_20261115/highprec.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    table(out)
    for c, v in out["mean"].items():
        print(c, {m: round(100 * x, 2) for m, x in v.items()})
    for k, v in out["comparisons"].items():
        print(k, {m: (round(v["delta_pp"][m], 2), [round(x, 2) for x in v["conditional_95_pp"][m]]) for m in METRICS})


if __name__ == "__main__":
    main()
