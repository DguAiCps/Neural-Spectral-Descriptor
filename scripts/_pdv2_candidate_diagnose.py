#!/usr/bin/env python3
"""선택 완료된 검증 결과에서 후보 선택과 수락 점수의 영향을 사후 분리한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_candidate_learning import OUT, CFG, VAL, OPS, BASE, FAMILY, read, setup
from _pdv2_no_context_ablation import write_json, sha256
from _xs_eval_lib import minmax
from _pr_eval_lib import classify_queries, pr_sweep
from retrieval.candidate_learning import CandidateScorer, pair_features


def main() -> None:
    """고정된 표현마다 네 가지 순위·판정 조합을 같은 원장에서 비교한다."""
    setup()
    summary = read(OUT / "summary.json")
    selection = read(BASE / "s1/confidence_selection.json")["ops"]
    result = {"scope": "선택 후 검증 원장의 사후 진단. 설정 재선택·추가 학습·시험 실행 없음.",
              "summary_sha256": sha256(OUT / "summary.json"), "arms": {},
              "script_sha256": sha256(Path(__file__))}
    for arm in CFG["arms"]:
        kind, mode = arm.split("_")
        epoch = summary["arms"][arm]["epoch"]
        tag = kind + "_initial" if mode == "head" else f"{arm}_epoch{epoch}"
        state = torch.load(OUT / arm / f"epoch{epoch}.pth", map_location="cuda", weights_only=False)
        heads = torch.nn.ModuleDict({op: CandidateScorer(tuple(CFG["head_hidden"])) for op in OPS}).cuda().eval()
        heads.load_state_dict(state["heads"], strict=True)
        sequences = {}
        for sensor, seq in VAL:
            path = OUT / "pools" / tag / "val" / f"{sensor}_{seq}"
            with np.load(path / "keys.npz") as z:
                emb = z["embedding"]
                embedding = torch.from_numpy(emb).cuda()
            sequences[sensor + "/" + seq] = {}
            for op in OPS:
                with np.load(path / (op + ".npz")) as z:
                    data = {k: z[k] for k in z.files}
                scores = []
                with torch.no_grad():
                    for start in range(0, len(data["queries"]), 128):
                        sl = slice(start, start + 128)
                        q, c, static = [torch.from_numpy(data[k][sl]).cuda() for k in ("queries", "candidates", "static")]
                        valid = c >= 0
                        features = pair_features(embedding, q, c, static.float(), valid)
                        scores.append(heads[op](features).masked_fill(~valid, -float("inf")).cpu().numpy())
                logits = np.concatenate(scores)
                indices = {"learned": logits.argmax(1), "fixed": np.zeros(len(logits), np.int64)}
                w = selection[op]["tuple"]
                for i, q in enumerate(data["queries"]):
                    valid = data["candidates"][i] >= 0
                    cand = data["candidates"][i, valid]
                    st = data["static"][i, valid].astype(np.float32)
                    score = (w[0] * minmax(emb[cand] @ emb[q]) + w[1] * minmax(st[:, 0].astype(np.float32))
                             + w[2] * minmax(st[:, 1]) + w[3] * minmax(st[:, 2]))
                    indices["fixed"][i] = int(np.argmax(score))
                rows = np.arange(len(logits))
                qc = classify_queries(data["min_past_gt_dist"])
                values = {}
                for rank, ix in indices.items():
                    for confidence in ("fixed", "learned"):
                        score = logits[rows, ix] if confidence == "learned" else data["static"][rows, ix, 2 if selection[op]["decision"] == "c_col" else 1]
                        distance = data["distances"][rows, ix]
                        metrics = None
                        if qc["n_pos"]:
                            sweep = pr_sweep(score, distance, qc)
                            metrics = {k: float(sweep[k]) for k in ("AP", "max_F1", "R@100P")}
                            metrics["R@1"] = float((distance[qc["positive"]] <= 5).mean())
                        values[rank + "_rank_" + confidence + "_confidence"] = metrics
                        if rank == confidence == "learned":
                            refpath = OUT / arm / f"val_epoch{epoch}" / f"{sensor}_{seq}_{op}.npz"
                            with np.load(refpath) as ref:
                                np.testing.assert_array_equal(data["candidates"][rows, ix], ref["j_star"])
                                np.testing.assert_allclose(score, ref["score_learned"], atol=2e-6, rtol=0)
                sequences[sensor + "/" + seq][op] = values
        macros = {}
        for op in OPS:
            macros[op] = {}
            for condition in next(iter(sequences.values()))[op]:
                groups = {}
                for sequence, values in sequences.items():
                    m = values[op][condition]
                    if m is not None:
                        sensor = sequence.split("/")[0]
                        groups.setdefault(FAMILY.get(sensor, sensor), []).append(m)
                macro = {k: float(np.mean([np.mean([m[k] for m in g]) for g in groups.values()]))
                         for k in ("AP", "max_F1", "R@100P", "R@1")}
                macros[op][condition] = macro
            for metric, value in summary["arms"][arm]["ops"][op].items():
                assert abs(macros[op]["learned_rank_learned_confidence"][metric] - value) < 1e-12
        result["arms"][arm] = {"epoch": epoch, "macro": macros, "sequences": sequences}
        write_json(OUT / "ranking_confidence_diagnosis.json", result)
        print("DIAGNOSED", arm, macros, flush=True)


if __name__ == "__main__":
    main()
