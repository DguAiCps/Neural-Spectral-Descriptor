#!/usr/bin/env python3
"""선형 융합의 원장·온라인 조건·상수5개 추론·정렬 캐시를 검사한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_linear_fusion import OUT, CFG, Scorer, base, fusion_features, ORIGINAL_BUILD_POOL
from _pdv2_no_context_ablation import write_json, sha256
from _xs_eval_lib import minmax
import _pdv2_candidate_verify as audit


def original_precision_minmax(values: np.ndarray) -> np.ndarray:
    """이미 확인한 기준 융합의 float32 계산을 그대로 재현한다."""
    assert np.array_equal(values, values.astype(np.float32))
    return minmax(values.astype(np.float32))


def main() -> None:
    """공통 감사와 모든 선택 원장의 가중합 내보내기 검사를 수행한다."""
    base.setup()
    audit.CandidateScorer, audit.pair_features = Scorer, fusion_features
    audit.minmax = original_precision_minmax
    audit.main()
    result = base.read(OUT / "completion_validation.json")
    result["status"] = "checking_export"
    write_json(OUT / "completion_validation.json", result)
    summary = base.read(OUT / "summary.json")
    exports, alignments = [], []
    for arm, row in summary["arms"].items():
        kind, mode = arm.split("_")
        epoch = row["epoch"]
        tag = kind + "_initial" if mode == "head" else f"{arm}_epoch{epoch}"
        state = torch.load(OUT / arm / f"epoch{epoch}.pth", map_location="cuda", weights_only=False)
        heads = torch.nn.ModuleDict({op: Scorer() for op in base.OPS}).cuda().eval()
        heads.load_state_dict(state["heads"], strict=True)
        assert all(sum(p.numel() for p in h.parameters()) == 5 for h in heads.values())
        for sensor, seq in base.VAL:
            folder = OUT / "pools" / tag / "val" / f"{sensor}_{seq}"
            with np.load(folder / "keys.npz") as z:
                embedding_np = z["embedding"]
            embedding = torch.from_numpy(embedding_np).cuda()
            if mode == "joint":
                _, _, spec, gt, _, _ = base.load_data("val", sensor + "/" + seq)
            for op in base.OPS:
                with np.load(folder / (op + ".npz")) as z:
                    data = {k: z[k] for k in ("queries", "candidates", "static")}
                params = row["exported_fusion"][op]
                weights = torch.tensor(params["weights"], device="cuda")
                bias = params["intercept"]
                chosen, scores, max_error = [], [], 0.
                with torch.no_grad():
                    for start in range(0, len(data["queries"]), 128):
                        sl = slice(start, start + 128)
                        q, c, static = [torch.from_numpy(data[k][sl]).cuda() for k in ("queries", "candidates", "static")]
                        valid = c >= 0
                        features = fusion_features(embedding, q, c, static.float(), valid)
                        direct = (features * weights).sum(-1) + bias
                        reference = heads[op](features)
                        error = float((direct - reference).abs().max())
                        assert error == 0., (arm, sensor, seq, op, error)
                        max_error = max(max_error, error)
                        value, index = direct.masked_fill(~valid, -float("inf")).max(1)
                        chosen.extend(c.gather(1, index[:, None]).squeeze(1).cpu().tolist())
                        scores.extend(value.cpu().tolist())
                with np.load(OUT / arm / f"val_epoch{epoch}" / f"{sensor}_{seq}_{op}.npz") as z:
                    np.testing.assert_array_equal(chosen, z["j_star"])
                    np.testing.assert_array_equal(scores, z["score_learned"])
                exports.append({"arm": arm, "sequence": sensor + "/" + seq, "op": op,
                                "queries": len(scores), "max_error": max_error})
                if mode == "joint":
                    ix = np.linspace(0, len(data["queries"]) - 1, 32, dtype=int)
                    fresh = ORIGINAL_BUILD_POOL(embedding_np, base.bank_for(op, spec), data["queries"][ix], gt[:, :3, 3])
                    np.testing.assert_array_equal(fresh["candidates"], data["candidates"][ix])
                    np.testing.assert_allclose(fresh["static"], data["static"][ix], atol=1e-6, rtol=1e-6)
                    alignments.append({"arm": arm, "sequence": sensor + "/" + seq, "op": op,
                                       "queries": len(ix), "max_error": float(np.abs(fresh["static"] - data["static"][ix]).max())})
    result.update(status="passed", exported_score_checks=exports, alignment_reuse_checks=alignments,
                  script_sha256=sha256(Path(__file__)),
                  precision_note="초기 기준 융합은 원래 float32로 계산. 원점수 가중합 내보내기는 평가와 같은 GPU 연산에서 전 선택 질의를 정확히 재현.")
    write_json(OUT / "completion_validation.json", result)
    print("LINEAR_VERIFIED", len(exports), len(alignments), flush=True)


if __name__ == "__main__":
    main()
