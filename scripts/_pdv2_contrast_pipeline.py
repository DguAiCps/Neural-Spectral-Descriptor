#!/usr/bin/env python3
"""고정 검색 연결의 전체·차이층·비차이층 학습을 세 시드에서 확인한다."""
import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import _pdv2_contrast_retrieval_canonical as previous

e, run, ROOT = previous.e, previous.run, previous.ROOT
PARENT = previous.OUT
OUT = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
SEEDS = previous.source.CAMPAIGN
SCOPE = ROOT / "results/pdv2_diffattn_contrast_scope_20261104"


def cases() -> dict:
    """선택된 전체 모델과 고정 대조를 재선택 없이 모은다."""
    result = {
        "identity": dict(kind="identity", seed=0, checkpoint=None),
        "fixed": dict(kind="fixed", seed=0, checkpoint=None),
    }
    selected = run.READ(SEEDS / "freeze.json")["selected_tags"]
    for seed in (1, 2, 3):
        for kind in ("diff", "abs_kv", "point_gate", "convs_only", "other_only"):
            if kind.endswith("only"):
                root = SCOPE if seed == 1 else SCOPE / f"s{seed}"
                checkpoint = root / kind / "epoch8.pth"
            else:
                root = SEEDS / f"s{seed}"
                arm, epoch = selected[kind].rsplit("_", 1)
                checkpoint = root / arm / f"epoch{epoch}.pth"
            result[f"{kind}_s{seed}"] = dict(
                kind=kind,
                seed=seed,
                checkpoint=str(checkpoint.relative_to(ROOT)),
                checkpoint_sha256=run.HASH(checkpoint),
            )
    return result


def prepare() -> dict:
    """첫 시드 검색 정책을 다른 시드·범위 대조에 고정한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = run.READ(path)
        for p, h in spec["sources"].items():
            assert run.HASH(ROOT / p) == h, p
    else:
        old = run.READ(PARENT / "protocol.json")
        selection = run.READ(PARENT / "summary.json")
        assert selection["selected_policy"] == "matching_only"
        candidate = run.READ(SEEDS / "summary.json")
        assert candidate["matching_candidate_passed"]
        spec = dict(
            created_unix=time.time(),
            train=old["train"],
            val=old["val"],
            confidence=old["confidence"],
            key_transform=old["key_transform"],
            cases=cases(),
            selected_policy="matching_only",
            protocol="두원래색인100+100과두코사인점수고정,두정렬점수에만학습복소차이출력사용. 최종시간판정도동일출력. raw키는스캔별직접정규화. 학습/체크포인트/검색정책재선택없음. 양의지표가없는KITTI09는계열평균에서제외.",
            criteria="3시드평균전체Diff가고정감산·absKV·점별게이트보다AP+.1%p,F1손실<=.3%p,R100손실<=1%p. 전체및convs_only의고정감산대비평균AP+.1%p와AP양수최소2/3. 동일시간BEV검증시드1참조대비AP/F1열세2%p,R100/R1열세3%p이내. 통과하면고정설정시험46·3시드로확인하며시험에서선택하지않음.",
            cache="원키·원보조색인과같은과거후보합집합/두코사인합만시퀀스당한번캐시한다. 각모델의두정렬점수는모든후보에서재계산한다. 원참조와전체첫시드원장일치·접두검색검사로동등성을확인한다.",
            scope="기존시험개발노출을알고있는탐색. 모델/정책선택은학습8·검증9만. 아직원고/그림변경없음.",
        )
        paths = [
            Path(__file__),
            Path(previous.__file__),
            PARENT / "protocol.json",
            PARENT / "summary.json",
            SEEDS / "summary.json",
            SEEDS / "integrated_validation.json",
            SCOPE / "three_seed_summary.json",
        ]
        paths += [
            ROOT / v["checkpoint"] for v in spec["cases"].values() if v["checkpoint"]
        ]
        spec["sources"] = {str(p.relative_to(ROOT)): run.HASH(p) for p in paths}
        run.WRITE(path, spec)
    run.OUT = OUT
    return spec


def raw_key(x: np.ndarray) -> np.ndarray:
    """고정 재계량 원키를 스캔별로 직접 계산한다."""
    mode = run.READ(OUT / "protocol.json")["key_transform"]
    raw = np.log1p(x) if mode == "log1p" else x.copy()
    raw = (
        raw.astype(np.float32)
        * np.load(run.source.OUT / "remetric_octave288.npy").astype(np.float32)[None]
    )
    return raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)


def candidate_pool(key: np.ndarray, bank: tuple, weights: np.ndarray) -> dict:
    """모델 공통의 실제 과거 합집합과 코사인 부분합을 보존한다."""
    queries = np.arange(30, len(key) - 15, dtype=np.int64)
    offsets = [0]
    ids_all = []
    static = []
    for q in queries:
        ids = np.unique(
            np.r_[
                run.base.prefix_topk(key, int(q), int(q) - 29, 100),
                run.base.prefix_topk(bank[0], int(q), int(q) - 29, 100),
            ]
        )
        score = weights[0] * run.base.minmax(key[ids] @ key[q]) + weights[
            1
        ] * run.base.minmax(bank[0][ids] @ bank[0][q])
        ids_all.append(ids)
        static.append(score)
        offsets.append(offsets[-1] + len(ids))
    return dict(
        query=queries,
        offset=np.asarray(offsets, np.int64),
        ids=np.concatenate(ids_all),
        static=np.concatenate(static).astype(np.float32),
    )


def winners(pool: dict, bank: tuple, weights: np.ndarray) -> np.ndarray:
    """후보별 새 정렬 점수를 모두 계산해 실제 최종 승자를 고른다."""
    chosen = []
    for n, q in enumerate(pool["query"]):
        lo, hi = pool["offset"][n : n + 2]
        ids = pool["ids"][lo:hi]
        a = run.base.align_scores(bank[1], bank[2], int(q), ids)
        c = run.base.column_align_scores(bank[3], int(q), ids)
        score = (
            pool["static"][lo:hi]
            + weights[2] * run.base.minmax(a)
            + weights[3] * run.base.minmax(c)
        )
        chosen.append(ids[np.argmax(score)])
    return np.asarray(chosen, np.int64)


def load_model(case: dict):
    """저장한 선택 상태를 그대로 불러온다."""
    if case["checkpoint"] is None:
        return None
    kind = "diff" if case["kind"] in ("convs_only", "other_only") else case["kind"]
    return previous.source.model(kind, ROOT / case["checkpoint"])


def transform(net, graph, kind: str) -> torch.Tensor:
    """학습 없음·고정 감산·학습 전달을 구별한다."""
    if kind == "identity":
        return graph.spectrum
    if kind == "fixed":
        return e.prior.filtered(graph, -0.5)
    return net(graph)


def worker(role: str, sequence: str) -> None:
    """공유 후보를 쓰되 모델별 실제 승자와 최종 점수를 다시 계산한다."""
    spec = prepare()
    assert role in ("val", "test") and sequence in spec[role]
    name = sequence.replace("/", "_")
    folder = OUT / role / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        return
    cache, pos, minimum, travel, bank, _, _, _ = run.inputs(role, sequence)
    graph, _, audit = e.prior.graph_inputs(role, sequence)
    raw = raw_key(cache["descriptors"])
    weights = run.cal.fusion_weights(spec["confidence"])[
        run.cal.FUSIONS.index("aux_half")
    ]
    poolpath = folder / "candidate_pool.npz"
    if poolpath.exists():
        pool = run.ARR(poolpath)
        assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    else:
        pool = candidate_pool(raw, bank, weights)
        np.savez_compressed(poolpath, **pool)
        run.WRITE(
            poolpath.with_suffix(".json"),
            dict(
                sha256=run.HASH(poolpath),
                queries=len(pool["query"]),
                pairs=len(pool["ids"]),
            ),
        )
    q = pool["query"]
    metrics = {}
    checks = []
    reproduction = {}
    with torch.no_grad():
        for tag, case in spec["cases"].items():
            path = folder / (tag + ".npz")
            if path.exists():
                rec = run.READ(path.with_suffix(".json"))
                assert run.HASH(path) == rec["sha256"]
                metrics[tag] = rec["metrics"]
                checks += rec["prefix_checks"]
                if "reproduction" in rec:
                    reproduction[tag] = rec["reproduction"]
                continue
            kind = case["kind"]
            net = load_model(case)
            z = transform(net, graph, kind)
            change = (
                bank
                if kind == "identity"
                else run.base.variant_inputs(z.cpu().numpy(), 8)
            )
            j = winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(
                query_idx=q,
                j_star=j,
                min_past_gt_dist=minimum[q],
                top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1),
                group=np.zeros(len(q), np.int64),
            )
            plan = e.prior.base.prior.plan(travel, q, j)
            values = []
            for start in range(0, len(q), 128):
                pp = {
                    k: torch.from_numpy(v[start : start + 128]).cuda()
                    for k, v in plan.items()
                }
                values.append(e.prior.score(z, pp).cpu().numpy())
            value = np.concatenate(values).astype(np.float64)
            metric = run.base.independent_metrics(row, value)
            other = run.white.prior.BASE.exp.metrics(value, row)
            for m in run.base.METRICS:
                if metric[m] is not None:
                    np.testing.assert_allclose(metric[m], other[m], atol=1e-12, rtol=0)
            prefix = []
            for at in (len(q) // 3, 2 * len(q) // 3):
                t = int(q[at])
                end = t + 16
                small, _, _ = e.prior.graph_inputs(role, sequence, end)
                short = transform(net, small, kind)
                skey = raw_key(cache["descriptors"][:end])
                np.testing.assert_array_equal(skey, raw[:end])
                short_bank = (
                    tuple(v[:end] for v in bank)
                    if kind == "identity"
                    else run.base.variant_inputs(short.cpu().numpy(), 8)
                )
                search_bank = (bank[0][:end], *short_bank[1:])
                winner = run.cal.candidates(skey, search_bank, weights[None], t)
                assert int(winner[0]) == int(j[at]), (tag, t, winner, j[at])
                pp = {
                    k: torch.from_numpy(v).cuda()
                    for k, v in e.prior.base.prior.plan(
                        travel[:end], np.array([t]), winner
                    ).items()
                }
                check = e.prior.score(short, pp).cpu().numpy()
                np.testing.assert_allclose(check, value[at : at + 1], atol=8e-6, rtol=0)
                prefix.append(
                    dict(
                        condition=tag,
                        query=t,
                        winner=int(winner[0]),
                        score_error=float(abs(check - value[at : at + 1]).max()),
                    )
                )
                del small, short
            rec = dict(
                metrics=metric,
                prefix_checks=prefix,
                checkpoint_sha256=case.get("checkpoint_sha256"),
            )
            if role == "val" and (
                tag == "fixed" or (case["seed"] == 1 and kind in previous.MODELS)
            ):
                ref = run.ARR(PARENT / "val" / name / (kind + "__matching_only.npz"))
                for k, v in row.items():
                    np.testing.assert_array_equal(v, ref[k])
                error = float(abs(value - ref["score"]).max())
                assert error < 8e-6
                rec["reproduction"] = dict(winners_exact=True, score_error=error)
                reproduction[tag] = rec["reproduction"]
            np.savez_compressed(path, **row, score=value)
            rec["sha256"] = run.HASH(path)
            run.WRITE(path.with_suffix(".json"), rec)
            metrics[tag] = metric
            checks += prefix
            print("EVALUATED", role, sequence, tag, metric, flush=True)
            del net, z, change
    run.WRITE(
        folder / "summary.json",
        dict(
            metrics=metrics,
            prefix_checks=checks,
            reproduction=reproduction,
            input_audit=audit,
            candidate_pool_sha256=run.HASH(poolpath),
        ),
    )


def finish() -> None:
    """실제 후보 선택 뒤 세 시드 지표와 시험 확장 기준을 계산한다."""
    spec = prepare()
    rows = {
        s: run.READ(OUT / "val" / s.replace("/", "_") / "summary.json")
        for s in spec["val"]
    }
    per = {
        tag: run.base.macro({s: r["metrics"][tag] for s, r in rows.items()})
        for tag in spec["cases"]
    }
    means = {}
    for kind in (
        "identity",
        "fixed",
        "diff",
        "abs_kv",
        "point_gate",
        "convs_only",
        "other_only",
    ):
        tags = (
            [kind]
            if kind in ("identity", "fixed")
            else [f"{kind}_s{s}" for s in (1, 2, 3)]
        )
        means[kind] = {
            m: float(np.mean([per[t]["macro"][m] for t in tags]))
            for m in run.base.METRICS
        }
    checks = {
        f"{c}_{m}": means["diff"][m] >= means[c][m] + d
        for c in ("fixed", "abs_kv", "point_gate")
        for m, d in (("AP", 0.001), ("max_F1", -0.003), ("R@100P", -0.01))
    }
    for kind in ("diff", "convs_only"):
        checks[kind + "_mean_gain"] = means[kind]["AP"] >= means["fixed"]["AP"] + 0.001
        checks[kind + "_two_seeds"] = (
            sum(
                per[f"{kind}_s{s}"]["macro"]["AP"] > means["fixed"]["AP"]
                for s in (1, 2, 3)
            )
            >= 2
        )
    bev = run.READ(
        ROOT / "results/pdv2_diffattn_comparable_20261104/bev_validation.json"
    )["temporal"]["macro"]
    checks.update(
        {
            "BEV_" + m: means["diff"][m] >= bev[m] - tol
            for m, tol in (
                ("AP", 0.02),
                ("max_F1", 0.02),
                ("R@100P", 0.03),
                ("R@1", 0.03),
            )
        }
    )
    report = dict(
        per_case=per,
        mean=means,
        checks=checks,
        advance_to_test=all(checks.values()),
        bev_temporal_validation_seed1=bev,
        test_used=False,
    )
    run.WRITE(OUT / "summary.json", report)
    print("PIPELINE", means, "ADVANCE", report["advance_to_test"], flush=True)


def campaign() -> None:
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    def call(s):
        with (logs / (s.replace("/", "_") + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, __file__, "val", "--sequence", s],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in spec["val"]]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            run.WRITE(
                OUT / "status.json",
                dict(phase="validation", done=i, total=len(futures)),
            )
            print("VALIDATED", i, flush=True)
    finish()
    run.WRITE(OUT / "status.json", dict(phase="complete", test_used=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "val", "finish", "run"))
    p.add_argument("--sequence", default="kitti/09")
    a = p.parse_args()
    e.configure()
    if a.action == "run":
        campaign()
    elif a.action == "val":
        worker("val", a.sequence)
    else:
        globals()[a.action]()
