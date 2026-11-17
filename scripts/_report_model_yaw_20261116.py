#!/usr/bin/env python3
"""새 보고모델(처음부터 학습, 시드 4·5·6, 융합 (2,1,2,2), lambda=0.25)로 질의 yaw 회전 분석을 다시 계산한다.

기존 분석(scripts/_review_direction_yaw.py, _review_direction_expand.py, _review_direction_causes.py,
_review_direction_expanded_report.py)의 질의 집합·각도·회전 규약·원시 재인코딩 결과를 그대로 쓰고
모델·융합 가중치·최종 판정 혼합만 새 보고모델로 바꾼다. 원시 회전 인코딩(288D 원키·BEV 복소계수)은
모델과 무관하므로 기존 expanded46/yaw/<seq>/encoded_*.npz를 해시 확인 뒤 읽기만 한다(심볼릭 링크).

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_report_model_yaw_20261116.py <stage>
stage: protocol | reproduce_old | baseline | rotate | causes | report | complete | all
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from pathlib import Path

for _n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_n] = "1"
sys.path[:0] = [str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _pdv2_scratch_search_20261116 as S  # noqa: E402
import _review_direction_yaw as yaw  # noqa: E402
import _review_direction_causes as causes  # noqa: E402
import _review_direction_expanded_report as rep  # noqa: E402
import _review_direction_complete as complete  # noqa: E402
from _pdv2_input_alignment_lib import pose_attributes, temporal_edges  # noqa: E402

P = S.P
base = yaw.report
B = yaw.current.run.base
ROOT = base.ROOT
OLD = base.OUT / "expanded46"
OUT = ROOT / "results/pdv2_report_model_20261116/yaw"
CFG = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
SEEDS = (4, 5, 6)
U = 512
TOL = 8e-6
ANGLES = yaw.ANGLES
CAUSE_ANGLES = causes.ANGLES
ARMS = causes.ARMS


def tag(seed: int) -> str:
    return f"nsd_s{seed}"


def ledger(sequence: str, seed: int) -> Path:
    return S.LEDGERS / "test" / sequence.replace("/", "_") / f"{S.model_tag(CFG, seed, U)}.npz"


def setting() -> tuple:
    w, lam = P.selected()
    assert w.tolist() == [2.0, 1.0, 2.0, 2.0] and lam == 0.25, (w, lam)
    return w, lam


# ----------------------------------------------------------------- 프로토콜
def protocol() -> dict:
    """기존 질의·각도·규약을 복사하고 모델·융합만 바꾼 새 프로토콜을 쓴다(있으면 해시 검사)."""
    path = OUT / "yaw_protocol.json"
    if path.exists():
        spec = base.read(path)
        for name, value in spec["sources"].items():
            assert base.digest(ROOT / name) == value, name
        return spec
    old = base.read(OLD / "yaw_protocol.json")
    w, lam = setting()
    cases = {}
    for s in SEEDS:
        ck = S.RUNS / S.canon(CFG) / f"s{s}" / f"u{U}.pth"
        cases[tag(s)] = dict(cfg=CFG, seed=s, updates=U, checkpoint=str(ck.relative_to(ROOT)), checkpoint_sha256=base.digest(ck),
                             main_ledger="results/pdv2_scratch_search_20261116/ledgers/test/<seq>/" + S.model_tag(CFG, s, U) + ".npz")
    paths = [Path(S.__file__), Path(P.__file__), Path(yaw.__file__), Path(causes.__file__), Path(rep.__file__),
             OLD / "yaw_protocol.json", OLD / "causes/protocol.json", P.OUT / "step1/selection.json"]
    paths += [ROOT / c["checkpoint"] for c in cases.values()]
    for seq in old["queries"]:
        folder = OLD / "yaw" / seq.replace("/", "_")
        for a in ANGLES:
            p = folder / f"encoded_{a:g}.npz"
            assert base.digest(p) == base.read(p.with_suffix(".json"))["sha256"]
            paths.append(p)
        for s in SEEDS:
            paths.append(ledger(seq, s))
    spec = dict(old)
    spec.update(created_unix=time.time(), cases=cases, fusion_weights=w.tolist(), score_lambda=lam,
        model="새 보고모델: 사전학습 없는 한 단계 학습, log1p 입력, LayerNorm, 게이트 고정 eta=-0.5, 512갱신. 구성 " + CFG,
        reuse="질의 집합(46시퀀스x64=2944)·각도·좌표 규약·원시 회전 인코딩(encoded_*.npz)은 기존 results/pdv2_review_direction_yaw_20261105/expanded46에서 그대로 가져온다. 인코딩은 모델과 무관하다. 질의 선정은 원장의 query_idx 축(모델과 무관한 고정 질의 배열 arange(30,N-15))의 등간격64개라 이전 모델 결과에 의존하지 않는다.",
        change="모델 S.load(CFG, seed, 512), 융합 가중치 P.selected()=(2,1,2,2) 그대로(S.pipe와 같음), 최종 판정 P.score(z, plan, 0.25). 융합 덧셈 순서는 pipeline.winners와 같게 정적 두 항을 float32로 먼저 더한다.",
        baseline="각도0에서2944질의x3시드의 승자 정확일치, 점수오차<=8e-6을 새 NSD 주 원장(results/pdv2_scratch_search_20261116/ledgers/test)과 검사.",
        parent_protocol_sha256=base.digest(OLD / "yaw_protocol.json"),
        sources={str(p.relative_to(ROOT)): base.digest(p) for p in paths})
    base.write(path, spec)
    oldc = base.read(OLD / "causes/protocol.json")
    cspec = dict(oldc)
    cspec.update(created_unix=time.time(), cases=cases, fusion_weights=w.tolist(), score_lambda=lam,
                 parent_protocol_sha256=base.digest(OLD / "causes/protocol.json"),
                 sources={str(p.relative_to(ROOT)): base.digest(p) for p in (Path(causes.__file__), path)})
    assert cspec["queries"] == {k: v[::4] for k, v in spec["queries"].items()}
    base.write(OUT / "causes/protocol.json", cspec)
    # 인코딩 링크(읽기 전용 원천)
    for seq in spec["queries"]:
        src, dst = OLD / "yaw" / seq.replace("/", "_"), OUT / "yaw" / seq.replace("/", "_")
        dst.mkdir(parents=True, exist_ok=True)
        for a in ANGLES:
            for ext in (".npz", ".json"):
                link = dst / f"encoded_{a:g}{ext}"
                if not link.exists():
                    link.symlink_to(os.path.relpath(src / link.name, dst))
    return spec


# ----------------------------------------------------------------- 질의 평가
@torch.no_grad()
def query(q, query_raw, query_z, query_coeff, keys, hkeys, gallery_z, travel, weights, lam) -> tuple:
    """yaw.query와 같은 독립 질의 검색이며 융합 순서·최종 판정만 새 경로(S.pipe)와 같게 한다."""
    saved_key, saved_hkey = keys[q].copy(), hkeys[q].copy()
    keys[q] = yaw.raw_key(query_raw[15:16])[0]
    hkeys[q] = B.variant_inputs(query_coeff[15:16], 8)[0][0]
    ids = np.unique(np.r_[B.prefix_topk(keys, q, q - 29, 100), B.prefix_topk(hkeys, q, q - 29, 100)])
    bank = B.variant_inputs(np.concatenate((gallery_z[ids], query_z[15:16]), 0), 8)
    local = np.arange(len(ids))
    a = B.align_scores(bank[1], bank[2], len(ids), local)
    c = B.column_align_scores(bank[3], len(ids), local)
    static = (weights[0] * B.minmax(keys[ids] @ keys[q]) + weights[1] * B.minmax(hkeys[ids] @ hkeys[q])).astype(np.float32)
    value = static + weights[2] * B.minmax(a) + weights[3] * B.minmax(c)
    j = int(ids[np.argmax(value)])
    keys[q], hkeys[q] = saved_key, saved_hkey
    plan = yaw.current.e.prior.base.prior.plan(travel, np.array([q]), np.array([j]))
    cq, cj = np.unique(plan["query"]), np.unique(plan["candidate"])
    assert cq.min() >= q - 5 and cq.max() <= q + 5
    compact = np.concatenate((query_z[cq - q + 15], gallery_z[cj]), 0)
    cp = dict(query=np.searchsorted(cq, plan["query"]), candidate=np.searchsorted(cj, plan["candidate"]) + len(cq), weight=plan["weight"])
    score = P.score(torch.from_numpy(compact).cuda(), {k: torch.from_numpy(v).cuda() for k, v in cp.items()}, lam)
    return j, float(score.cpu().numpy()[0]), len(ids), int(j + 20 >= q - 15)


def encoded(sequence: str, angles) -> dict:
    folder = OUT / "yaw" / sequence.replace("/", "_")
    out = {}
    for a in angles:
        p = folder / f"encoded_{a:g}.npz"
        assert base.digest(p) == base.read(p.with_suffix(".json"))["sha256"]
        out[a] = dict(np.load(p))
    return out


def ledgers_for(sequence, models, weights, lam, angles, queries, folder=None, compare=None) -> dict:
    """모델별·각도별 원장을 계산한다. models: {tag: net}. compare(tag, angle) -> 참조 원장 dict(또는 None)."""
    enc = encoded(sequence, angles)
    cache, z, travel = yaw.original_inputs(sequence)
    keys, hkeys = yaw.raw_key(cache["descriptors"]), B.variant_inputs(z, 8)[0]
    assert np.array_equal(keys, yaw.current.raw_key(cache["descriptors"]))  # 새 경로 원키와 같음
    queries = np.asarray(queries, np.int64)
    out = {}
    for t, net in models.items():
        gallery_z = yaw.infer(net, cache["descriptors"], z, cache["poses"], cache["poses"][0, :3, :3])
        for angle in angles:
            raw = enc[angle]
            vals = []
            for q in queries:
                take = np.searchsorted(raw["frames"], np.arange(q - 15, q + 16))
                poses = cache["poses"][q - 15:q + 16].copy()
                poses[:, :3, :3] = poses[:, :3, :3] @ yaw.rotz(angle).T
                local_z = yaw.infer(net, raw["raw"][take], raw["coeff"][take], poses, cache["poses"][0, :3, :3])
                vals.append(query(int(q), raw["raw"][take], local_z, raw["coeff"][take], keys, hkeys, gallery_z, travel, weights, lam))
            ref = compare(t, angle)
            refix = np.searchsorted(ref["query_idx"], queries) if ref is not None else None
            row = dict(query_idx=queries, j_star=np.asarray([v[0] for v in vals], np.int64), score=np.asarray([v[1] for v in vals]),
                       min_past_gt_dist=None, top1_geo_dist=None,
                       n_candidates=np.asarray([v[2] for v in vals]), gallery_raw_support_overlaps_query=np.asarray([v[3] for v in vals]))
            out[(t, angle)] = (row, ref, refix, cache)
        del gallery_z
    return out


def reproduce_old() -> None:
    """같은 코드 경로에 이전 모델(diff_s1)·이전 융합을 넣어 기존 expanded46 원장 일부를 재현한다."""
    spec = protocol()
    seq = "kitti/00"
    folder = OLD / "yaw" / seq.replace("/", "_")
    old_w = yaw.current.run.cal.fusion_weights(base.read(base.MAIN / "protocol.json")["confidence"])[yaw.current.run.cal.FUSIONS.index("aux_half")]
    net = yaw.current.load_model(base.read(OLD / "yaw_protocol.json")["cases"]["diff_s1"])
    tick = time.time()
    res = ledgers_for(seq, {"diff_s1": net}, old_w, 0.25, (0., 90., 7.3), spec["queries"][seq],
                      compare=lambda t, a: dict(np.load(folder / f"{t}_{a:g}.npz")))
    rec = {}
    for (t, a), (row, ref, _, _) in res.items():
        rec[f"{t}_{a:g}"] = dict(winners_equal=bool(np.array_equal(row["j_star"], ref["j_star"])),
                                 score_bitwise_equal=bool(np.array_equal(row["score"], ref["score"])),
                                 score_max_error=float(np.max(abs(row["score"] - ref["score"]))),
                                 n_candidates_equal=bool(np.array_equal(row["n_candidates"], ref["n_candidates"])))
    ok = all(v["winners_equal"] and v["score_max_error"] <= TOL for v in rec.values())
    base.write(OUT / "old_model_reproduction.json", dict(status="passed" if ok else "failed", sequence=seq, model="diff_s1(이전 보고모델)",
               fusion_weights=np.asarray(old_w).tolist(), records=rec, seconds=time.time() - tick,
               note="새 스크립트의 query()(pipeline.winners 융합 순서, P.score lambda=0.25)에 이전 모델·이전 융합을 넣은 재현."))
    print("REPRO_OLD", ok, rec, flush=True)
    assert ok


def run_stage(stage: str) -> None:
    spec = protocol()
    w, lam = setting()
    angles = (0.,) if stage == "baseline" else tuple(ANGLES[1:])
    if stage == "rotate":
        assert base.read(OUT / "yaw_baseline_validation.json")["status"] == "passed"
    for seq in spec["queries"]:
        folder = OUT / "yaw" / seq.replace("/", "_")
        done = folder / f"{stage}.json"
        if done.exists():
            continue
        tick = time.time()
        models = {tag(s): S.load(CFG, s, U) for s in SEEDS}
        mains = {tag(s): dict(np.load(ledger(seq, s))) for s in SEEDS}
        res = ledgers_for(seq, models, w, lam, angles, spec["queries"][seq], compare=lambda t, a: mains[t])
        records = {}
        for (t, a), (row, ref, refix, cache) in res.items():
            q = row["query_idx"]
            row["min_past_gt_dist"] = ref["min_past_gt_dist"][refix]
            row["top1_geo_dist"] = np.linalg.norm(cache["poses_gt"][q, :3, 3] - cache["poses_gt"][row["j_star"], :3, 3], axis=1)
            np.testing.assert_array_equal(ref["query_idx"][refix], q)
            path = folder / f"{t}_{a:g}.npz"
            np.savez_compressed(path, **row)
            rec = dict(sha256=base.digest(path), angle=a, metrics=base.metric(row["score"], row["top1_geo_dist"], row["min_past_gt_dist"]))
            if a == 0:
                enc0 = encoded(seq, (0.,))[0.]
                z = yaw.original_inputs(seq)[1]
                rec["baseline_check"] = dict(winner_mismatches=int(np.sum(row["j_star"] != ref["j_star"][refix])),
                    score_max_error=float(np.max(abs(row["score"] - ref["score"][refix]))), score_tolerance=TOL,
                    geo_dist_max_error=float(np.max(abs(row["top1_geo_dist"] - ref["top1_geo_dist"][refix]))),
                    raw_descriptor_max_error=float(np.max(abs(enc0["raw"] - cache["descriptors"][enc0["frames"]]))),
                    raw_coefficient_max_error=float(np.max(abs(enc0["coeff"] - z[enc0["frames"]]))))
                base.write(path.with_suffix(".json"), rec)
                assert rec["baseline_check"]["winner_mismatches"] == 0 and rec["baseline_check"]["score_max_error"] <= TOL, (seq, t, rec["baseline_check"])
            base.write(path.with_suffix(".json"), rec)
            records[f"{t}_{a:g}"] = rec
        base.write(done, dict(status="passed", records=records, seconds=time.time() - tick))
        print(stage.upper(), seq, round(time.time() - tick, 1), flush=True)
        del models
    if stage == "baseline":
        rows = {s: base.read(OUT / "yaw" / s.replace("/", "_") / "baseline.json") for s in spec["queries"]}
        checks = [r["baseline_check"] for v in rows.values() for r in v["records"].values()]
        assert len(checks) == 46 * 3 and all(r["winner_mismatches"] == 0 and r["score_max_error"] <= TOL for r in checks)
        base.write(OUT / "yaw_baseline_validation.json", dict(status="passed", queries=2944, query_seed_checks=8832, winner_mismatches=0,
                   max_score_error=max(r["score_max_error"] for r in checks), tolerance=TOL,
                   reference="results/pdv2_scratch_search_20261116/ledgers/test/<seq>/" + S.model_tag(CFG, 0, U).replace("@s0@", "@s{4,5,6}@") + ".npz",
                   sequences=rows))


# ----------------------------------------------------------------- 원인 대조
def final_scores_new(q, ids, qz, gallery, travel) -> np.ndarray:
    """causes.final_scores와 같되 최종 판정을 P.score(lambda)로 한다."""
    plan = yaw.current.e.prior.base.prior.plan(travel, np.full(len(ids), q, np.int64), np.asarray(ids, np.int64))
    qi, ji = np.unique(plan["query"]), np.unique(plan["candidate"])
    compact = np.concatenate((qz[qi - q + 15], gallery[ji]))
    pp = dict(query=np.searchsorted(qi, plan["query"]), candidate=np.searchsorted(ji, plan["candidate"]) + len(qi), weight=plan["weight"])
    with torch.no_grad():
        return P.score(torch.from_numpy(compact).cuda(), {k: torch.from_numpy(v).cuda() for k, v in pp.items()}, LAM).cpu().numpy()


def fusion_new(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """pipeline.winners와 같은 순서: 정적 두 항을 float32로 먼저 더한다."""
    static = (weights[0] * values[:, 0] + weights[1] * values[:, 1]).astype(np.float32)
    return static + weights[2] * values[:, 2] + weights[3] * values[:, 3]


LAM = 0.25


def causes_sequence(sequence: str, cspec: dict, w: np.ndarray) -> None:
    """causes.run_sequence와 같은 대조를 새 모델로 실행한다(모델·융합·판정 외 동일)."""
    name = sequence.replace("/", "_")
    target = OUT / "causes" / (name + ".json")
    if target.exists():
        return
    folder = OUT / "yaw" / name
    enc = encoded(sequence, (0., *CAUSE_ANGLES))
    cache, z, travel = yaw.original_inputs(sequence)
    keys, hkeys = yaw.raw_key(cache["descriptors"]), B.variant_inputs(z, 8)[0]
    rows, checks = [], []
    started = time.time()
    for t, case in cspec["cases"].items():
        net = S.load(CFG, case["seed"], U)
        gallery = yaw.infer(net, cache["descriptors"], z, cache["poses"], cache["poses"][0, :3, :3])
        actual = {a: dict(np.load(folder / f"{t}_{a:g}.npz")) for a in (0., *CAUSE_ANGLES)}
        for q in cspec["queries"][sequence]:
            take = np.searchsorted(enc[0.]["frames"], np.arange(q - 15, q + 16))
            x0, z0 = enc[0.]["raw"][take], enc[0.]["coeff"][take]
            p0 = cache["poses"][q - 15:q + 16].copy(); first = cache["poses"][0, :3, :3]
            r0 = first.T[None] @ p0[:, :3, :3]
            yaw0 = np.arctan2(r0[:, 1, 0], r0[:, 0, 0])
            baseoutput = causes.infer(net, x0, z0, p0, first)
            ref = causes.evaluate(q, x0, z0, baseoutput, keys, hkeys, gallery, travel, w)
            ix = np.searchsorted(actual[0.]["query_idx"], q)
            assert ref["winner"] == actual[0.]["j_star"][ix]
            assert abs(ref["confidence"] - actual[0.]["score"][ix]) <= TOL
            for angle in CAUSE_ANGLES:
                take = np.searchsorted(enc[angle]["frames"], np.arange(q - 15, q + 16))
                xa, za = enc[angle]["raw"][take], enc[angle]["coeff"][take]
                phase = np.exp(-1j * np.deg2rad(angle) * np.arange(1, 9))[None, None].astype(np.complex64)
                zi = z0 * phase
                pa = p0.copy(); pa[:, :3, :3] = pa[:, :3, :3] @ yaw.rotz(angle).T
                rel = first.T[None] @ pa[:, :3, :3]
                ya = np.arctan2(rel[:, 1, 0], rel[:, 0, 0]); expected_yaw = yaw0 - np.deg2rad(angle)
                residual = np.rad2deg(np.angle(np.exp(1j * (ya - expected_yaw))))
                edges = temporal_edges(np.zeros(31, np.int64))
                geom_error = float(np.max(abs(pose_attributes(pa, edges) - pose_attributes(p0, edges))))
                for arm in ARMS:
                    ideal_grid = arm in ("ideal_grid", "ideal_both", "ideal_both_exact_geometry", "direct_output_phase")
                    x, coeff = (x0, zi) if ideal_grid else (xa, za)
                    override = expected_yaw if arm in ("ideal_transport", "ideal_both", "ideal_both_exact_geometry") else None
                    poses = p0 if arm == "ideal_both_exact_geometry" else pa
                    output = baseoutput * phase if arm == "direct_output_phase" else causes.infer(net, x, coeff, poses, first, override)
                    value = causes.evaluate(q, x, coeff, output, keys, hkeys, gallery, travel, w, ref)
                    if arm == "actual":
                        ix = np.searchsorted(actual[angle]["query_idx"], q)
                        assert value["winner"] == actual[angle]["j_star"][ix], (sequence, t, q, angle)
                        error = abs(value["confidence"] - actual[angle]["score"][ix])
                        assert error <= TOL, (sequence, t, q, angle, error)
                        checks.append(error)
                    output_error = np.linalg.norm((causes.unit(output[10:21]) - causes.unit(baseoutput[10:21] * phase)).reshape(11, -1), axis=1)
                    rows.append(dict(sequence=sequence, seed=int(case["seed"]), query=int(q), angle=angle, arm=arm,
                        input_key_l2=float(np.linalg.norm(yaw.raw_key(x[15:16]) - yaw.raw_key(x0[15:16]))),
                        input_coefficient_phase_l2=float(np.linalg.norm(causes.unit(coeff[15:16]) - causes.unit(zi[15:16]))),
                        output_phase_l2_max=float(output_error.max()), yaw_phase_residual_deg_max=float(abs(residual).max()), geometry_attribute_error=geom_error,
                        candidate_symmetric_difference=int(len(np.setxor1d(value["ids"], ref["ids"]))),
                        key_candidates_changed=int(not np.array_equal(value["keyids"], ref["keyids"])),
                        auxiliary_candidates_changed=int(not np.array_equal(value["auxids"], ref["auxids"])),
                        raw_channel_max_error=np.max(abs(value["raw"] - ref["raw"]), axis=0).tolist(),
                        normalized_channel_max_error=np.max(abs(value["norm"] - ref["norm"]), axis=0).tolist(),
                        baseline_channel_range=np.ptp(ref["raw"], axis=0).tolist(),
                        fixed_pool_fusion_max_error=float(np.max(abs(value["fused"] - ref["fused"]))),
                        baseline_margin=ref["margin"], baseline_candidates=len(ref["ids"]), winner_changed=bool(value["winner"] != ref["winner"]),
                        fixed_pool_winner_changed=bool(value["fixed_winner"] != ref["winner"]),
                        winner=value["winner"], baseline_winner=ref["winner"],
                        confidence_change=value["confidence"] - ref["confidence"], fixed_winner_confidence_change=value["fixed_confidence"] - ref["confidence"]))
        del net, gallery
    base.write(target, dict(status="passed", sequence=sequence, rows=rows, actual_ledger_checks=len(checks), actual_max_score_error=max(checks),
               seconds=time.time() - started, protocol_sha256=base.digest(OUT / "causes/protocol.json")))
    print("CAUSE", sequence, len(rows), round(time.time() - started, 1), flush=True)


def run_causes() -> None:
    protocol()
    cspec = base.read(OUT / "causes/protocol.json")
    for name, value in cspec["sources"].items():
        assert base.digest(ROOT / name) == value, name
    w, lam = setting()
    global LAM
    LAM = lam
    causes.final_scores = final_scores_new
    causes.fusion = fusion_new
    for seq in cspec["queries"]:
        causes_sequence(seq, cspec, w)


# ----------------------------------------------------------------- 집계·완료 검산
def report() -> None:
    """기존 집계 함수를 새 폴더에 그대로 적용한다(첫 시드 태그만 바꿈)."""
    spec = protocol()
    first = next(iter(spec["cases"]))
    src = inspect.getsource(rep.collect_yaw)
    assert src.count("tag=='diff_s1'") == 1
    namespace = dict(vars(rep))
    exec(compile(src.replace("tag=='diff_s1'", f"tag=={first!r}"), rep.__file__, "exec"), namespace)
    rep.OUT = namespace["OUT"] = OUT
    namespace["collect_yaw"]()
    rep.collect_causes()
    print("REPORT passed", flush=True)


def finish() -> None:
    """원장·지표·인코딩·원인 대조를 다시 읽어 검산한다(_review_direction_complete.py의 yaw 관련 부분)."""
    spec = protocol()
    summary = base.read(OUT / "yaw_summary.json")
    cz = base.read(OUT / "causes/summary.json")
    assert summary["status"] == cz["status"] == "passed"
    ledger_count = rows = encoding_count = baseline_rows = 0
    independent_error = baseline_error = 0.
    for seq, queries in spec["queries"].items():
        folder = OUT / "yaw" / seq.replace("/", "_")
        expected_frames = np.unique(np.concatenate([np.arange(q - 15, q + 16) for q in queries]))
        assert len(expected_frames) == spec["support_frames"][seq]
        assert base.read(folder / "baseline.json")["status"] == base.read(folder / "rotate.json")["status"] == "passed"
        for angle in spec["angles"]:
            enc = folder / f"encoded_{angle:g}.npz"
            assert base.digest(enc) == base.read(enc.with_suffix(".json"))["sha256"]
            with np.load(enc) as data:
                np.testing.assert_array_equal(data["frames"], expected_frames)
            encoding_count += 1
            for t, case in spec["cases"].items():
                data = dict(np.load(folder / f"{t}_{angle:g}.npz"))
                assert len(data["query_idx"]) == 64
                actual = complete.independent_metrics(data)
                expected = summary["final_recognition"][str(angle)]["per_seed"][t]["sequences"][seq]
                if actual is None:
                    assert expected is None
                else:
                    for key in base.KEYS:
                        error = abs(actual[key] - expected[key])
                        independent_error = max(error, independent_error)
                        assert error <= 1e-12
                if angle == 0:
                    original = dict(np.load(ledger(seq, case["seed"])))
                    ix = np.searchsorted(original["query_idx"], data["query_idx"])
                    np.testing.assert_array_equal(data["j_star"], original["j_star"][ix])
                    error = float(np.max(abs(data["score"] - original["score"][ix])))
                    assert error <= TOL
                    baseline_error = max(error, baseline_error)
                    baseline_rows += 64
                ledger_count += 1
                rows += 64
    assert (ledger_count, rows, encoding_count, baseline_rows) == (966, 61824, 322, 8832)
    assert cz["rows"] == 39744 and cz["actual_ledger_checks"] == 6624
    # 독립 집계: 시퀀스->6도시->시드 평균을 위 독립 지표로 다시 계산
    macro_check = {}
    for angle in spec["angles"]:
        vals = {k: [] for k in base.KEYS}
        for t in spec["cases"]:
            per = {}
            for seq in spec["queries"]:
                per[seq] = complete.independent_metrics(dict(np.load(OUT / "yaw" / seq.replace("/", "_") / f"{t}_{angle:g}.npz")))
            cities = [[per[s] for s in seqs if per[s] is not None] for seqs in base.CITY.values()]
            assert len(cities) == 6 and all(cities)
            for k in base.KEYS:
                vals[k].append(np.mean([np.mean([m[k] for m in ms]) for ms in cities]))
        err = max(abs(np.mean(vals[k]) - summary["final_recognition"][str(angle)]["mean_std"][k]["mean"]) for k in base.KEYS)
        macro_check[str(angle)] = err
        assert err <= 1e-12, (angle, err)
    repro = base.read(OUT / "old_model_reproduction.json")
    assert repro["status"] == "passed"
    files = [OUT / "yaw_summary.json", OUT / "yaw_metrics.csv", OUT / "yaw_city_dataset.csv", OUT / "causes/summary.json",
             OUT / "yaw_baseline_validation.json", OUT / "old_model_reproduction.json", Path(__file__)]
    base.write(OUT / "completion_validation.json", dict(status="passed", completed_unix=time.time(), sequences=46, queries=2944,
        ledger_files=ledger_count, ledger_rows=rows, encoding_archives=encoding_count,
        raw_support_frames=spec["total_raw_support_frames"],
        baseline_query_seed_checks=baseline_rows, baseline_winner_mismatches=0, baseline_max_score_error=baseline_error,
        independent_metric_max_error=independent_error, independent_macro_max_error=macro_check,
        independent_metric_scope="966원장의64질의마다 모든 고유 임계값을 직접 수락한 PR곡선(_review_direction_complete.independent_metrics)과 집계JSON 비교, 시퀀스->6도시->시드 평균도 따로 재계산.",
        cause_rows=cz["rows"], cause_actual_ledger_checks=cz["actual_ledger_checks"], cause_actual_max_score_error=cz["actual_max_score_error"],
        old_model_reproduction=repro["records"],
        scope_note="기존 completion_validation.json의 방향 학습·roll·격자 검사는 모델 교체와 무관해 이 단위에서 다시 하지 않았다(원래 파일 참조).",
        artifacts_sha256={str(p.relative_to(ROOT)): base.digest(p) for p in files}))
    print("COMPLETION PASSED", ledger_count, rows, independent_error, macro_check, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("protocol", "reproduce_old", "baseline", "rotate", "causes", "report", "complete", "all"))
    a = parser.parse_args()
    P.e.configure()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tick = time.time()
    stages = ("protocol", "reproduce_old", "baseline", "rotate", "causes", "report", "complete") if a.stage == "all" else (a.stage,)
    for st in stages:
        t0 = time.time()
        {"protocol": protocol, "reproduce_old": reproduce_old, "baseline": lambda: run_stage("baseline"),
         "rotate": lambda: run_stage("rotate"), "causes": run_causes, "report": report, "complete": finish}[st]()
        print("STAGE", st, round(time.time() - t0, 1), flush=True)
    print("TOTAL", round(time.time() - tick, 1), flush=True)


if __name__ == "__main__":
    main()
