#!/usr/bin/env python3
"""시퀀스 검증과 고정 시험의 비교·비용·학습 표본을 함께 마감한다."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO),str(REPO/"src"),str(REPO/"scripts")]
import _pdv2_sequence_fixed_test as test
import _pdv2_spectral_evidence as exp
from _pdv2_no_context_ablation import sha256, write_json
from _pr_eval_lib import classify_queries, operating_point

OUT = REPO/"results/pdv2_sequence_report_20261018"


def main() -> None:
    """산출물과 같은 시드 효과를 확인한 뒤 그림·전체 표를 저장한다."""
    freeze = test.check_freeze()
    val = exp.read(test.VAL/"summary.json")
    result = exp.read(test.OUT/"summary.json")
    for root in (test.VAL,test.OUT):
        assert exp.read(root/"status.json")["status"]=="complete"
        assert exp.read(root/"completion_validation.json")["status"]=="passed"
    for p,h in exp.read(test.VAL/"protocol.json")["runtime_sha256"].items():
        assert sha256(REPO/p)==h,p
    assert exp.read(REPO/"results/pdv2_sequence_target_20261018/target_audit.json")["summary_sha256"] == result["baseline_summary_sha256"]
    all_pairs = {}
    for seed in ("1","2","3"):
        tau = freeze["seeds"][seed]["new_tau"]["tau"]
        groups = {}
        for family in ("kitti","helipr","mulran"):
            f1s = []
            for sensor,seq in exp.old.base.VAL:
                sequence = sensor+"/"+seq
                path = test.VAL/f"s{seed}/standard_{sensor}_{seq}.npz"
                a = exp.arrays(path)
                qc = classify_queries(a["min_past_gt_dist"])
                if sequence.startswith(family) and qc["n_pos"]:
                    f1s.append(operating_point(a[test.METHOD],a["top1_geo_dist"],qc,tau)["F1"])
                row = exp.read(test.VAL/f"{sensor}_{seq}.json")["results"][seed]["standard"]
                if family=="kitti":
                    for method,c in row["pair_counts"].items():
                        all_pairs.setdefault(method,[]).append((len(a["query_idx"]),c))
            groups[family] = float(np.mean(f1s))
        assert abs(np.mean(list(groups.values()))-freeze["seeds"][seed]["new_tau"]["validation_macro_F1"]) < 1e-12
    cost = {m:{"mean_unique_pairs":sum(n*c["pairs_mean"] for n,c in rows)/sum(n for n,c in rows),
               "max_unique_pairs":max(c["pairs_max"] for n,c in rows),
               "effective_neighbors_mean":sum(n*c["effective_neighbors_mean"] for n,c in rows)/sum(n for n,c in rows)}
            for m,rows in all_pairs.items()}
    train = exp.read(REPO/"results/pdv2_training_coverage_20261018/summary.json")
    for p,h in train["sources"].items():
        assert sha256(REPO/p)==h,p
    # 공식 비교 방법의 원시 점수에서 공통 질의 지표를 다시 계산한다.
    bp = exp.read(REPO/"results/pdv2_main_official_baselines/baselines_common.json")["methods"]["bevplace"]
    for sensor,seq in test.TEST:
        a = exp.arrays(test.OUT/"test"/f"s1/{sensor}_{seq}.npz")
        b = exp.arrays(REPO/f"results/pdv2_bevplace_official/odom/dump_bevplace_{sensor}_{seq}.npz")
        ix = np.searchsorted(b["query_idx"],a["query_idx"])
        calculated = exp.metrics(b["score_negl2"][ix],{k:b[k][ix] for k in ("min_past_gt_dist","top1_geo_dist")})
        if calculated["n_pos"]:
            for metric in exp.METRICS:
                assert abs(calculated[metric]-bp["per_sequence"][sensor+"/"+seq][metric]) < 1e-12
    per_city = {c:{m:100*(np.mean([result["per_seed"][s][test.METHOD]["cities"][c][m] for s in ("1","2","3")])
                            -bp["city_groups"][c][m]) for m in exp.METRICS} for c in test.CITY}
    paired = {m:[100*(result["per_seed"][s][test.METHOD]["macro"][m]-result["per_seed"][s]["raw"]["macro"][m])
                  for s in ("1","2","3")] for m in exp.METRICS}
    fixed_tau = {}
    for seed in ("1", "2", "3"):
        tau = freeze["seeds"][seed]["new_tau"]["tau"]
        records = {}
        totals = {key: 0 for key in ("TP", "FP", "FN", "TN")}
        for sensor, seq in test.TEST:
            sequence = sensor + "/" + seq
            a = exp.arrays(test.OUT / "test" / f"s{seed}/{sensor}_{seq}.npz")
            qc = classify_queries(a["min_past_gt_dist"])
            measured = operating_point(a[test.METHOD], a["top1_geo_dist"], qc, tau)
            saved = exp.read(test.OUT / "test" / f"{sensor}_{seq}.json")
            assert measured == saved["results"][seed]["fixed_tau"]
            records[sequence] = {"n_pos": qc["n_pos"], **measured}
            for key in totals:
                totals[key] += measured[key]
        cities = {}
        for city, members in test.CITY.items():
            chosen = [records[seq] for seq in members if records[seq]["n_pos"]]
            cities[city] = {
                metric: float(np.mean([row[metric] for row in chosen]))
                for metric in ("precision", "recall", "F1")
            }
        fixed_tau[seed] = {
            "tau": tau,
            "cities": cities,
            "macro": {
                metric: float(np.mean([row[metric] for row in cities.values()]))
                for metric in ("precision", "recall", "F1")
            },
            "all_46_counts": totals,
        }
    summary = {"validation":val,"fixed_test":result,"test_delta_vs_raw_pp":paired,
               "city_delta_vs_bevplace_pp":per_city,"pair_cost":cost,"training_coverage":train,
               "fixed_validation_tau_test": fixed_tau,
               "scope":"시험 결과로 재선택하지 않는다. 비용은 최종 후보 주변 고유 스펙트럼 쌍 수이며 전체 로봇 지연 측정이 아니다."}
    write_json(OUT/"summary.json",summary)
    with (OUT/"validation_all.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["model","method","metric","mean_percent","sample_sd_percent","seed1","seed2","seed3"])
        for model,methods in val["mean_std"].items():
            for method,row in methods.items():
                for metric in exp.METRICS:
                    values=[val["per_seed"][s][model][method][metric]*100 for s in ("1","2","3")]
                    assert abs(np.mean(values)-row[metric]["mean"]*100)<1e-12
                    assert abs(np.std(values,ddof=1)-row[metric]["std"]*100)<1e-12
                    w.writerow([model,method,metric,row[metric]["mean"]*100,row[metric]["std"]*100,*values])
    with (OUT/"fixed_test.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["method","metric","mean_percent","sample_sd_percent","seed1","seed2","seed3"])
        for method,row in result["mean_std"].items():
            for metric in exp.METRICS:
                values=[result["per_seed"][s][method]["macro"][metric]*100 for s in ("1","2","3")]
                np.testing.assert_allclose([np.mean(values),np.std(values,ddof=1)],
                    [row[metric]["mean"]*100,row[metric]["std"]*100],atol=1e-12,rtol=0)
                w.writerow([method,metric,row[metric]["mean"]*100,row[metric]["std"]*100,*values])
        for metric in exp.METRICS:
            w.writerow(["BEVPlace++ seed1",metric,bp["macro"][metric]*100,"","","",""])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(OUT/"UnDotum.ttf"))
    plt.rcParams.update({"font.family":"UnDotum","axes.unicode_minus":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.7))
    labels=["같은 시간 모델\n단일 쌍 판정","같은 시간 모델\n거리 두 이웃","고정한 새 NSD\n거리 열 이웃·중복 보정","BEVPlace++\n공식 시드 1"]
    for ax,metric in zip(axes,("AP","R@100P")):
        means=[result["mean_std"][m][metric]["mean"]*100 for m in test.SCORES]+[bp["macro"][metric]*100]
        std=[result["mean_std"][m][metric]["std"]*100 for m in test.SCORES]+[0.]
        bars=ax.bar(np.arange(4),means,color=["#9aa2aa","#70a6d1","#276fbf","#df873d"])
        ax.errorbar(np.arange(3), means[:3], yerr=std[:3], fmt="none", color="black", capsize=3)
        ax.bar_label(bars,fmt="%.2f",padding=4,fontsize=9)
        ax.set_xticks(np.arange(4),labels,fontsize=9)
        ax.set_ylim(0, max(means)*1.15)
        ax.set_ylabel(metric+" (%)")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("고정 시험 46시퀀스 · 공통 195,726질의 · 6도시 평균",fontsize=13)
    fig.text(.5,.015,"NSD는 3시드 평균±표본 표준편차. 검증에서 한 조건을 고정했으며 시험으로 재선택하지 않았다.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.95))
    for ext in ("png","pdf"):
        fig.savefig(OUT/("fixed_test."+ext),dpi=180)
    plt.close(fig)
    write_json(OUT/"completion_validation.json",{"status":"passed","script_sha256":sha256(Path(__file__)),
        "validation_tau_recomputed":True,"fixed_tau_test_ledgers_recomputed":138,
        "bevplace_common_metrics_recomputed":True,"source_frozen":True,
        "artifacts":{str(p.relative_to(REPO)):sha256(p) for p in OUT.iterdir()
                     if p.suffix in (".csv",".png",".pdf")}})
    print("FINAL",result["mean_std"][test.METHOD],result["delta_vs_bevplace_pp"],"COST",cost,flush=True)


if __name__=="__main__":
    main()
