#!/usr/bin/env python3
"""고정된 학습 질의에서 정렬 점수와 키의 순위 교정 가능 범위를 진단한다."""
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_fixed_fusion_20261015"


def main() -> None:
    """기존 후보 안에서 키 항이 뒤집을 수 없는 정렬 점수 차이를 센다."""
    protocol = json.loads((OUT / "protocol.json").read_text())
    roster = protocol["training_roster"]["queries"]
    result = {"scope": "고정 학습 질의의 초기 후보 사후 진단. 설정·질의·학습·선택을 변경하지 않음.",
              "meaning": "정답의 최대 정적점수+키가중치 < 오답의 최대 정적점수이면, 정규화 키 점수를 [0,1] 안에서 어떻게 바꿔도 이 고정 후보 목록에서는 정답을 1등으로 만들 수 없다. 후보 갱신 효과는 이 한계에 포함하지 않음.",
              "arms": {}}
    for kind in ("gnn", "point"):
        result["arms"][kind] = {}
        for op, setting in protocol["score"].items():
            aggregate = {"n": 0, "key_top1_correct": 0, "fusion_top1_correct": 0, "unrepairable_by_key_alone": 0, "fusion_loss_sum": 0., "unrepairable_loss_sum": 0.}
            sequences = {}
            for sequence, queries in roster.items():
                if not queries:
                    continue
                folder = OUT / "pools" / f"{kind}_initial/train" / sequence.replace("/", "_")
                with np.load(folder / "keys.npz") as z:
                    embedding = z["embedding"]
                with np.load(folder / (op + ".npz")) as z:
                    data = {k: z[k] for k in ("queries", "candidates", "static", "correct")}
                row = dict.fromkeys(aggregate, 0)
                for i in np.searchsorted(data["queries"], queries):
                    valid = data["candidates"][i] >= 0
                    c = data["candidates"][i, valid]
                    correct = data["correct"][i, valid]
                    key = embedding[c] @ embedding[data["queries"][i]]
                    key = (key - key.min()) / max(float(key.max()) - float(key.min()), 1e-8)
                    static = data["static"][i, valid].astype(np.float32)
                    static = (static - static.min(0)) / np.maximum(static.max(0) - static.min(0), 1e-8)
                    weights = np.asarray(setting["tuple"], dtype=np.float32)
                    offset = static @ weights[1:]
                    fused = weights[0] * key
                    for channel in range(3):
                        fused = fused + weights[channel + 1] * static[:, channel]
                    row["n"] += 1
                    row["key_top1_correct"] += int(correct[np.argmax(key)])
                    row["fusion_top1_correct"] += int(correct[np.argmax(fused)])
                    impossible = offset[correct].max() + weights[0] + 1e-6 < offset[~correct].max()
                    row["unrepairable_by_key_alone"] += int(impossible)
                    logits = fused.astype(np.float64) / weights[0] / protocol["config"]["temperature"]
                    logits -= logits.max()
                    loss = float(np.log(np.exp(logits).sum()) - np.log(np.exp(logits[correct]).sum()))
                    row["fusion_loss_sum"] += loss
                    row["unrepairable_loss_sum"] += loss * int(impossible)
                sequences[sequence] = row
                for key, value in row.items():
                    aggregate[key] += value
            result["arms"][kind][op] = {"aggregate": aggregate, "sequences": sequences}
    directions = {"scope": "고정 학습 질의의 방향 구성. 양성 쌍의 방향 빈도와 구분한다.",
                  "labels": {"0": "같은 방향 정답 있음", "1": "반대 방향 정답만 있음", "2": "그 외"},
                  "total": {str(i): 0 for i in range(3)}, "sequences": {}}
    for sequence, queries in roster.items():
        folder = OUT / "pools" / "gnn_initial/train" / sequence.replace("/", "_")
        with np.load(folder / "pbev128_main.npz") as z:
            groups = z["group"][np.searchsorted(z["queries"], queries)]
        row = {str(i): int((groups == i).sum()) for i in range(3)}
        directions["sequences"][sequence] = row
        for key, count in row.items():
            directions["total"][key] += count
    directions["protocol_sha256"] = hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest()
    directions["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "training_directions.json").write_text(json.dumps(directions, ensure_ascii=False, indent=2) + "\n")
    result["protocol_sha256"] = hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "training_score_diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print({kind: {op: row["aggregate"] for op, row in ops.items()} for kind, ops in result["arms"].items()})


if __name__ == "__main__":
    main()
