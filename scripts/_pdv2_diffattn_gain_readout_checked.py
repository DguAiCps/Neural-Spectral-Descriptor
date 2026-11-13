#!/usr/bin/env python3
"""정규화 전 출력의 크기와 구분하여 실제 미래 의존성과 최종 점수를 검사한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_gain_readout as e
from retrieval.context_spectral_gain import spectral_gain


def prepare() -> dict:
    """검사 교체 이유와 실패 로그를 보존하며 학습·점수식은 유지한다."""
    spec = e.prepare()
    target = e.OUT / "validation_runtime_amendment.json"
    if not target.exists():
        e.run.WRITE(
            target,
            dict(
                created_unix=time.time(),
                source_sha256=e.run.HASH(Path(__file__)),
                protocol_sha256=e.run.HASH(e.OUT / "protocol.json"),
                diagnostic_sha256=e.run.HASH(e.OUT / "prefix_diagnostic.json"),
                failures={
                    str(p.relative_to(e.ROOT)): e.run.HASH(p)
                    for p in (e.OUT / "logs").glob("val_*.log")
                },
                reason="정규화전출력을단위키허용오차로검사해7시퀀스중단. 별도진단에서그래프없는모델도절단시logit최대.000733차이·같은크기미래제거0차이. 표준attention도같은입력반복반올림오차가존재. 미래제거와실제판정점수를별도로검사. 모델/학습/점수식/선택기준변경없음.",
                verification="원키288블록의기존허용오차유지. 정규화전128값의차이는기록. 실제양의이득은maxabs<=.002검사. 실제접두판정점수atol8e-6기존기준유지. 같은노드/간선배열크기를유지하며미래입력0·미래접촉간선을마지막더미노드로격리한추가검사도판정atol8e-6. 두지점검사.",
                output="val_checked/ 아래모든시퀀스재평가. 이전val/와실패로그보존. 비공개시험사용없음.",
            ),
        )
    else:
        assert e.run.READ(target)["source_sha256"] == e.run.HASH(Path(__file__))
    return spec


def evaluate(sequence: str) -> None:
    """같은 점수식을 평가하고 실제 절단·같은 크기 미래 제거를 비교한다."""
    prepare()
    name = sequence.replace("/", "_")
    folder = e.OUT / "val_checked"
    folder.mkdir(exist_ok=True)
    target = folder / (name + ".json")
    if target.exists():
        return
    cache, _, _, travel, _, _, _, paths = e.run.inputs("val", sequence)
    data = e.run.ARR(e.OUT / "pools/val" / name / "pool.npz")
    row = {
        k: data[k]
        for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")
    }
    z = e.prior.coefficients(paths[1])
    p = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
    graph = e.run.make_graph(
        cache["descriptors"], cache["poses"], np.zeros(len(cache["poses"]), np.int64)
    ).cuda()
    cases = [("initial", "global", None, 0.5), ("uniform", "global", None, 0)] + [
        (f"{a}_{ep}", k, e.OUT / a / f"epoch{ep}.pth", 0.5)
        for a, (k, _) in e.ARMS.items()
        for ep in e.EPOCHS
    ]
    values = {}
    metrics = {}
    checks = []
    for tag, kind, cp, strength in cases:
        net = e.model(kind, cp)
        with torch.no_grad():
            key = net(graph)
            parts = []
            for start in range(0, len(row["query_idx"]), 128):
                pp = {
                    k: torch.from_numpy(v[start : start + 128]).cuda()
                    for k, v in p.items()
                }
                parts.append(e.score(key, z, pp, strength).cpu().numpy())
            value = np.concatenate(parts).astype(np.float64)
            for at in (len(row["query_idx"]) // 3, 2 * len(row["query_idx"]) // 3):
                t = int(row["query_idx"][at])
                end = t + 16
                small = e.run.make_graph(
                    cache["descriptors"][:end],
                    cache["poses"][:end],
                    np.zeros(end, np.int64),
                ).cuda()
                short = net(small)
                np.testing.assert_allclose(
                    short[: t + 6, :288].cpu(),
                    key[: t + 6, :288].cpu(),
                    atol=4e-6,
                    rtol=4e-5,
                )
                pad = graph.clone()
                pad.x[end:] = 0
                invalid = (pad.edge_index[0] >= end) | (pad.edge_index[1] >= end)
                pad.edge_index[:, invalid] = len(pad.x) - 1
                pad.edge_attr[invalid] = 0
                masked = net(pad)

                def error(a, b):
                    return float((a - b).abs().max())

                gain = spectral_gain(key[: t + 6], strength)
                gain_error = error(spectral_gain(short[: t + 6], strength), gain)
                masked_gain_error = error(
                    spectral_gain(masked[: t + 6], strength), gain
                )
                assert gain_error <= 0.002 and masked_gain_error <= 0.002
                pp = {
                    k: torch.from_numpy(v).cuda()
                    for k, v in e.prior.plan(
                        travel[:end],
                        row["query_idx"][at : at + 1],
                        row["j_star"][at : at + 1],
                    ).items()
                }
                for k, v in pp.items():
                    np.testing.assert_array_equal(v.cpu(), p[k][at : at + 1])
                cut_value = e.score(short, z[:end], pp, strength).cpu().numpy()
                pad_value = e.score(masked, z, pp, strength).cpu().numpy()
                np.testing.assert_allclose(
                    cut_value, value[at : at + 1], atol=8e-6, rtol=0
                )
                np.testing.assert_allclose(
                    pad_value, value[at : at + 1], atol=8e-6, rtol=0
                )
                checks.append(
                    dict(
                        condition=tag,
                        query=t,
                        gain_error=gain_error,
                        masked_gain_error=masked_gain_error,
                        logit_error=error(short[: t + 6, 288:], key[: t + 6, 288:]),
                        score_error=float(abs(cut_value - value[at : at + 1]).max()),
                        masked_score_error=float(
                            abs(pad_value - value[at : at + 1]).max()
                        ),
                    )
                )
                del small, short, pad, masked
        values[tag] = value
        metrics[tag] = e.run.base.independent_metrics(row, value)
        other = e.run.white.prior.BASE.exp.metrics(value, row)
        for m in e.run.base.METRICS:
            if metrics[tag][m] is not None:
                np.testing.assert_allclose(
                    other[m], metrics[tag][m], atol=1e-12, rtol=0
                )
        del net, key
    path = target.with_suffix(".npz")
    np.savez_compressed(path, **row, **values)
    e.run.WRITE(
        target,
        dict(
            status="passed",
            sequence=sequence,
            metrics=metrics,
            prefix_checks=checks,
            ledger_sha256=e.run.HASH(path),
            amendment_sha256=e.run.HASH(e.OUT / "validation_runtime_amendment.json"),
            test_used=False,
        ),
    )
    print("VAL_CHECKED", sequence, flush=True)


def finish() -> None:
    """동결된 원래 선택 기준을 새 검사 완료 원장에 그대로 적용한다."""
    spec = prepare()
    rec = {
        s: e.run.READ(e.OUT / "val_checked" / (s.replace("/", "_") + ".json"))
        for s in spec["val"]
    }
    tags = next(iter(rec.values()))["metrics"]
    results = {
        k: e.run.base.macro({s: r["metrics"][k] for s, r in rec.items()}) for k in tags
    }
    best = {
        kind: max(
            [
                f"{a}_{ep}"
                for a, (k, _) in e.ARMS.items()
                if k == kind
                for ep in e.EPOCHS
            ],
            key=lambda tag: results[tag]["macro"]["AP"],
        )
        for kind in e.KINDS
    }
    candidate = best["diff"]
    m = results[candidate]["macro"]
    controls = ["initial"] + [best[k] for k in e.KINDS if k != "diff"]
    checks = {f"AP_{c}": m["AP"] >= results[c]["macro"]["AP"] + 0.001 for c in controls}
    checks.update(
        {
            f"{metric}_{c}": m[metric] >= results[c]["macro"][metric] - tol
            for c in controls
            for metric, tol in (("max_F1", 0.003), ("R@100P", 0.01))
        }
    )
    bev = e.run.READ(e.prior.previous.prior.gate.previous.OUT / "bev_validation.json")[
        "temporal"
    ]["macro"]
    checks.update(
        {
            f"bev_{k}": m[k] >= bev[k] - tol
            for k, tol in (
                ("AP", 0.02),
                ("max_F1", 0.02),
                ("R@100P", 0.03),
                ("R@1", 0.03),
            )
        }
    )
    selected = candidate if all(checks.values()) else None
    e.run.WRITE(
        e.OUT / "summary.json",
        dict(
            results=results,
            best=best,
            checks=checks,
            selected=selected,
            bev_temporal=bev,
            ledger_root="val_checked",
            amendment_sha256=e.run.HASH(e.OUT / "validation_runtime_amendment.json"),
            test_used=False,
        ),
    )
    print(
        "FINISHED",
        selected,
        {k: results[v]["macro"] for k, v in best.items()},
        flush=True,
    )


def campaign() -> None:
    """학습을 다시 하지 않고 독립 시퀀스 최종 검사를 완료한다."""
    spec = prepare()
    logs = e.OUT / "logs_checked"
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
            e.run.WRITE(
                e.OUT / "status.json",
                dict(phase="checked_validation", done=i, total=len(futures)),
            )
            print("CHECKED", i, len(futures), flush=True)
    finish()
    e.run.WRITE(e.OUT / "status.json", dict(phase="complete", test_used=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("val", "finish", "run"))
    parser.add_argument("--sequence", default="kitti/09")
    args = parser.parse_args()
    e.run.configure()
    if args.action == "val":
        evaluate(args.sequence)
    elif args.action == "run":
        campaign()
    else:
        finish()
