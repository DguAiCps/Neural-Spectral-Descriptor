#!/usr/bin/env python3
"""단위 G 기준 원장 재현: S.pipe를 그대로 쓰되 원장 폴더만 임시 위치로 바꿔 보고모델 시드 4(와 uniform)를 다시 만들고
기존 원장(results/pdv2_scratch_search_20261116/ledgers)과 승자·점수·지표가 비트 단위로 같은지 본다. 기존 원장은 읽기만 한다.

docker exec -w <repo> ring_baseline python3 scripts/_report_model_gate_repro_20261116.py <role> <출력 폴더> <모델>[,<모델>...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _pdv2_scratch_search_20261116 as S  # noqa: E402

role, out, models = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(",")
S.P.e.configure()
torch.set_num_threads(4)
ref = S.LEDGERS
S.LEDGERS = out / "ledgers"
seqs = S.VAL if role == "val" else S.TESTSEQ
res = {}
for s in seqs:
    S.pipe(role, s, models, 3)
    name = s.replace("/", "_")
    for m in models:
        a, b = np.load(S.LEDGERS / role / name / (m + ".npz")), np.load(ref / role / name / (m + ".npz"))
        ja, jb = json.loads((S.LEDGERS / role / name / (m + ".json")).read_text()), json.loads((ref / role / name / (m + ".json")).read_text())
        res.setdefault(m, {})[s] = dict(
            winners_equal=bool(np.array_equal(a["j_star"], b["j_star"])), query_equal=bool(np.array_equal(a["query_idx"], b["query_idx"])),
            score_bitwise_equal=bool(np.array_equal(a["score"], b["score"])), metrics_equal=ja["metrics"] == jb["metrics"])
summary = {m: dict(all_equal=all(all(x.values()) for x in v.values()), n=len(v), sequences=v) for m, v in res.items()}
(out / f"repro_{role}.json").write_text(json.dumps(summary, indent=1))
print({m: (v["all_equal"], v["n"]) for m, v in summary.items()})
