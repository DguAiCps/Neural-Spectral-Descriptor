#!/usr/bin/env python3
"""공통 검증으로 선택된 SeqOT의 native 예측을 독립 집계한다."""
import json
from pathlib import Path

import numpy as np

from _review_native_seqot_cpu_gem import BASE, REPO, sha, write

OUT = BASE / "common_selection_native"


def main() -> None:
    """저장된 순위와 공개 정답만으로 hit 수와 3시드 평균을 검산한다."""
    protocol = json.loads((BASE / "protocol.json").read_text())
    labels = np.load(REPO / protocol["ground_truth"], allow_pickle=True)
    positive = np.asarray([len(g) > 0 for g in labels])
    assert positive.sum() == 5336
    manifest = json.loads((OUT / "selection_manifest.json").read_text())
    rows = []
    for candidate in manifest["candidates"]:
        root = OUT / candidate["name"]
        report = json.loads((root / "report.json").read_text())
        path = root / "predicted_L2_dis.npz"
        assert sha(path) == report["prediction_sha256"]
        pred = np.load(path)["arr_0"].reshape(-1, 22, 3)
        np.testing.assert_array_equal(pred[:, 0, 0], np.arange(0, 5648 * 5, 5))
        ids = pred[:, :, 1].astype(np.int64)
        counts = {str(k): sum(bool(np.isin(ids[i, :k], labels[i]).any()) for i in np.flatnonzero(positive)) for k in (1, 5, 20)}
        assert counts == report["hits"]
        recall = {k: v / int(positive.sum()) for k, v in counts.items()}
        assert recall == report["official_recall"]
        rows.append({"candidate": candidate, "hits": counts, "recall": recall, "prediction_sha256": sha(path), "passed": True})
    groups = {}
    for objective in ("AP", "R@1"):
        chosen = [r for r in rows if objective in r["candidate"]["selection_objectives"]]
        assert sorted(r["candidate"]["seed"] for r in chosen) == [1, 2, 3]
        groups[objective] = {k: {"mean": float(np.mean([r["recall"][k] for r in chosen])), "sample_std": float(np.std([r["recall"][k] for r in chosen], ddof=1))} for k in ("1", "5", "20")}
    original = [json.loads((BASE / f"v2_s{s}/native/report.json").read_text())["official_recall"] for s in (1, 2, 3)]
    groups["original"] = {k: {"mean": float(np.mean([r[k] for r in original])), "sample_std": float(np.std([r[k] for r in original], ddof=1))} for k in ("1", "5", "20")}
    write(OUT / "independent_verification.json", {"complete": True, "rows": rows, "groups": groups,
          "positive": int(positive.sum()), "selection_manifest_sha256": sha(OUT / "selection_manifest.json"),
          "scope": "독립 공개15m 정답 검산; 검증선택의 native 일반화 진단. 원 논문 overlap>.3 표와 직접 동일 조건이 아니며 native 수치로 재선택하지 않음."})
    lines = [r"\paragraph{Checkpoint selection and SeqOT native retrieval.}",
             r"After freezing checkpoint selection on the common validation split (99 retained stage-2 checkpoints per seed), we evaluate the six AP- and R@1-selected checkpoints on the same native NCLT features and released labels. Stage 1 remains fixed. The released labels use a 15\,m spatial threshold rather than the paper's overlap threshold, so these are implementation-level diagnostics rather than an exact reproduction of the published table.",
             r"\begin{center}\begin{tabular}{lrrr}", r"Validation criterion & R@1 & R@5 & R@20 \\", r"\hline"]
    for key, title in [("original", "Original choice"), ("AP", "Common AP"), ("R@1", "Common R@1")]:
        cells = [f"${100*groups[key][k]['mean']:.2f}\\pm{100*groups[key][k]['sample_std']:.2f}$" for k in ("1", "5", "20")]
        lines.append(title + " & " + " & ".join(cells) + r" \\")
    lines += [r"\end{tabular}\end{center}", r"Entries report mean and sample standard deviation over three seeds, on the same 5,336 positive queries. Selection uses validation data only; these native scores do not determine checkpoint choice. CPU evaluation preserves all top-1 matches and R@1/5/20 on the three original GPU controls; its maximum descriptor difference is $2.98\times10^{-8}$. The official GeM code and raw L2 retrieval are retained."]
    (BASE.parent / "paper_proposal/seqot_selection_addendum.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(groups, ensure_ascii=False))


if __name__ == "__main__":
    main()
