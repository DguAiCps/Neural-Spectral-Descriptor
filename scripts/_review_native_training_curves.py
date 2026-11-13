#!/usr/bin/env python3
"""보존한 학습·검증 기록만으로 수렴 진단 도표를 만든다."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105/training_curves"


def main() -> None:
    """동일 축의 선은 같은 지표이며 시험 성능은 사용하지 않는다."""
    OUT.mkdir(exist_ok=True)
    font = FontProperties(fname="/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes = axes.flatten()
    colors = ["#0072B2", "#D55E00", "#009E73"]
    records, sources = [], {}
    for seed in [1, 2, 3]:
        suffix = "" if seed == 1 else f"_s{seed}"
        paths = {"HeLiOS": f"results/pdv2_helios_official{suffix}/train_record.json",
                 "ImLPR": f"results/pdv2_imlpr_official{suffix}/train_selection.json",
                 "SeqOT": f"results/pdv2_seqot_official_centered{suffix}/train_selection.json"}
        data = {}
        for method, relative in paths.items():
            content = (REPO / relative).read_bytes()
            sources[relative] = hashlib.sha256(content).hexdigest()
            data[method] = json.loads(content)
        series = [
            (0, "HeLiOS", "학습 AP", "--", [(r["epoch"], r["train_ap"]) for r in data["HeLiOS"]["epochs"]]),
            (0, "HeLiOS", "검증 AP", "-", [(r["epoch"], r["val_ap"]) for r in data["HeLiOS"]["epochs"]]),
            (1, "ImLPR", "검증 R@1", "-", [(r["trained_epochs"], r["family_macro_r1"]) for r in data["ImLPR"]["evals"]]),
            (2, "SeqOT stage1", "검증 R@1", "-", [(int(k), r["family_r1_l2"]) for k, r in data["SeqOT"]["stage1"]["epochs"].items()]),
            (3, "SeqOT stage2", "검증 R@1", "-", [(int(k), r["family_r1_l2"]) for k, r in data["SeqOT"]["stage2"]["epochs"].items()]),
        ]
        for panel, method, metric, style, values in series:
            values.sort()
            x, y = zip(*values)
            axes[panel].plot(x, [v * 100 for v in y], style, color=colors[seed - 1], linewidth=1.2,
                             alpha=0.65 if style == "--" else 1)
            records.extend({"method": method, "seed": seed, "metric": metric, "completed_epoch": e,
                            "value": value} for e, value in values)
    titles = ["HeLiOS: 학습·검증 AP", "ImLPR: 검증 R@1", "SeqOT 특징 모델: 검증 R@1", "SeqOT GeM: 검증 R@1"]
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("완료한 학습 에포크")
        axis.set_ylabel("지표 (%)")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(handles=[Line2D([0], [0], color=color, label=f"시드 {i + 1}") for i, color in enumerate(colors)],
                    loc="best", frameon=False, fontsize=8)
    axes[0].text(0.98, 0.03, "실선: 검증 / 점선: 학습", transform=axes[0].transAxes, ha="right", fontsize=9)
    fig.suptitle("보존 학습 기록의 진단: 시험 지표를 사용하지 않음", fontsize=13)
    fig.savefig(OUT / "retained_training_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "retained_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (OUT / "retained_training_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "provenance.json").write_text(json.dumps({"sources_sha256": sources, "rows": len(records),
        "scope": "보존된 원 학습 검증 규약의 곡선이며 새 공통 KISS 검증 99,236질의의 선택 결과가 아니다. ImLPR의 에포크 시작 평가를 완료 에포크로 표시한다. 지표 간 절대값을 비교하지 않는다.",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, ensure_ascii=False, indent=2) + "\n")
    print({"rows": len(records), "output": str(OUT)})


if __name__ == "__main__":
    main()
