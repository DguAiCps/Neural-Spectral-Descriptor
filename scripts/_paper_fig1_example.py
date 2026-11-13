#!/usr/bin/env python3
"""그림 1용 실제 사례 수치: 한 질의와 같은 장소·닮은 장소 참조의 크기 키, 위상 정렬 곡선, 수락 점수.

컨테이너 ring_baseline 안에서 실행한다. 결과는 /workspace/data/nsd_results/fig1_example_20261111/에 둔다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
OUT = Path("/workspace/data/nsd_results/fig1_example_20261111")
SEQ = "mulran/DCC01"


def main() -> None:
    import _pdv2_contrast_pipeline_test as T

    P = T.pipeline
    P.e.configure()
    spec = T.activate()
    cache, pos, minimum, travel, bank, _, _, _ = P.run.inputs("test", SEQ)
    graph, _, _ = P.e.prior.graph_inputs("test", SEQ)
    key = P.raw_key(cache["descriptors"])
    key = key / np.linalg.norm(key, axis=1, keepdims=True)
    mag = np.abs(graph.spectrum.cpu().numpy()).reshape(len(key), -1)
    mag = mag / np.linalg.norm(mag, axis=1, keepdims=True)
    with torch.no_grad():
        z0 = graph.spectrum.cpu().numpy()
        z1 = P.transform(P.load_model(spec["cases"]["diff_s1"]), graph, "diff").cpu().numpy()
    np.savez_compressed(OUT / "features.npz", key=key, mag=mag, z0=z0, z1=z1, pos=pos, minimum=minimum, travel=travel)
    print(key.shape, mag.shape, z0.shape, z1.shape, pos.shape, sorted(cache.keys()))

    # 사례 규칙: 재방문 질의 중 크기 키 1위가 200 m 넘게 먼 장소이고, NSD가 실제로 5 m 안의 장소를 반환해
    # 이 시퀀스의 100% 정밀도 임계값을 넘긴 질의 가운데, 키 1위 장소와의 BEV 크기 유사도가 가장 낮은 것을 쓴다.
    # A는 NSD가 실제로 반환한 장소, B는 크기 키 1위다. 보기 쉬운 사례를 고르는 규칙이며 대표 표본이 아니다.
    with np.load(T.OUT / "test" / SEQ.replace("/", "_") / "diff_s1.npz") as z:
        led = {k: z[k] for k in z.files}
    wrong = ((led["min_past_gt_dist"] <= 5) & (led["top1_geo_dist"] > 5)) | (led["min_past_gt_dist"] > 25)
    threshold = float(led["score"][wrong].max())
    rows, aliased = [], 0
    for i, t in enumerate(led["query_idx"]):
        if led["min_past_gt_dist"][i] > 5 or led["top1_geo_dist"][i] > 5:
            continue
        cand = np.arange(0, t - 29)
        b = cand[(key[cand] @ key[t]).argmax()]
        far = float(np.linalg.norm(pos[b] - pos[t]))
        aliased += far > 25
        if far > 200 and led["score"][i] > threshold:
            rows.append((float(mag[t] @ mag[b]), int(t), int(led["j_star"][i]), int(b), float(led["score"][i])))
    _, q, a, b, score = min(rows)
    from _pd_build_test_caches import _make_loader
    sensor, seq = SEQ.split("/")
    loader = _make_loader(sensor, seq)
    ids = cache["scan_ids"] if "scan_ids" in cache else cache["frame_indices"]
    points = {n: np.asarray(loader[int(ids[k])]["points"], np.float32)[:, :3] for n, k in (("q", q), ("a", a), ("b", b))}
    np.savez_compressed(OUT / "example_points.npz", **points)
    (OUT / "example.json").write_text(json.dumps(dict(
        sequence=SEQ, rule="재방문 질의 중 크기 키 1위가 200 m 초과이고 NSD(diff_s1)가 5 m 이내 장소를 반환해 시퀀스의 100% 정밀도 임계값을 넘긴 질의 가운데, 키 1위 장소와의 BEV 크기 유사도가 최소인 것",
        eligible=len(rows), key_top1_distant_and_nsd_correct=int(aliased), threshold=threshold,
        revisit_queries=int((led["min_past_gt_dist"] <= 5).sum()),
        bev_cos_returned=float(mag[q] @ mag[a]), bev_cos_key_top1=float(mag[q] @ mag[b]),
        query=q, returned=a, key_top1=b, acceptance_score=score,
        dist_returned=float(np.linalg.norm(pos[a] - pos[q])), dist_key_top1=float(np.linalg.norm(pos[b] - pos[q])),
        key_cos_returned=float(key[q] @ key[a]), key_cos_top1=float(key[q] @ key[b])), ensure_ascii=False, indent=1))
    print((OUT / "example.json").read_text())


if __name__ == "__main__":
    main()
