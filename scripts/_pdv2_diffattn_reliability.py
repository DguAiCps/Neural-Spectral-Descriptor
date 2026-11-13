#!/usr/bin/env python3
"""DiffAttnConv 문맥을 시간 정렬 증거의 신뢰도에 직접 연결해 학습한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_diffattn_alignment_teacher as prior
from retrieval.context_reliability import reliability_score
from retrieval.sequence_evidence import neighbor_plans, repeated_candidate_weights
from retrieval.spectral_evidence import correlation_profiles

run = prior.run
PARENT = prior.PARENT
OUT = ROOT / "results/pdv2_diffattn_reliability_20261104"
MODES = ("attenuate", "reweight")
GAMMAS = (0.0, 0.25, 1.0, 4.0)
ARMS = {
    f"{m}_{tag}": (m, lr) for m in MODES for tag, lr in (("lr5", 1e-5), ("lr4", 1e-4))
}
EPOCHS = (4, 8)


def prepare() -> dict:
    """새 검증 전에 가설·대조·학습 범위와 확장 기준을 기록한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = run.READ(path)
        for name, digest in spec["sources"].items():
            assert run.HASH(ROOT / name) == digest, name
        return spec
    base = run.READ(PARENT / "protocol.json")
    initial = PARENT / "recenter_frozen/epoch0.pth"
    sources = [
        Path(__file__),
        ROOT / "src/retrieval/context_reliability.py",
        ROOT / "tests/retrieval/test_context_reliability.py",
        ROOT / "src/retrieval/sequence_evidence.py",
        ROOT / "src/retrieval/spectral_evidence.py",
        ROOT / "src/gnn/model.py",
        ROOT / "src/gnn/geometry_interaction.py",
        Path(run.__file__),
        Path(run.source.__file__),
        Path(run.current.adoption.__file__),
        ROOT / "scripts/_pdv2_context_accept_lib.py",
        initial,
        prior.OUT / "summary.json",
        prior.BASELINE / "val/summary.json",
    ]
    sources += [
        PARENT / "pools" / s.replace("/", "_") / "graph.pt" for s in base["train"]
    ]
    spec = dict(
        created_unix=time.time(),
        seed=1,
        initial=str(initial.relative_to(ROOT)),
        train=base["train"],
        val=base["val"],
        confidence=base["confidence"],
        modes=MODES,
        gammas=GAMMAS,
        arms=ARMS,
        epochs=EPOCHS,
        training_gamma=1.0,
        steps_per_sequence=8,
        batch_queries=128,
        weight_decay=1e-5,
        clip=1.0,
        architecture="기존288+128·시간±5·2층 차이K/V 및 기하 결합. gnn.convs.*14텐서만 갱신. 투영·게이트가중치·BN통계·재계량 고정. 게이트출력간접변화는허용. 새학습모듈없음.",
        score="각 대응쌍의 게이트문맥 벡터 거리d=||alpha_i*c_i-alpha_j*c_j||²/2. r=exp(-gamma*d). attenuate는기존시간가중치에r을곱함. reweight는그가중치의합을1로재정규화. 두방향최대는기존대로4스펙트럼채널별. 원본75/보정25·크기제곱근·60각도·두채널척도평균 유지. gamma0기존스펙트럼판정재현. 가산문맥점수없음.",
        data="학습8의초기포화완화모델로실제100+100검색승자를새로생성. 양성minpast<=5·음성>25·회색질의제외. 정답은승자거리<=5. 시퀀스별같은배치·8epoch512업데이트. 양성/음성승자최대64씩균등추출,한쪽없으면다른쪽128. 학습승자는고정,검증은각키의후보/승자를전부재검색. GT학습/KISS검증.",
        calibration="학습8초기승자에서각규칙gamma1의양성/음성균등로지스틱적합. 이척도로최종BCE를두층에역전파. 초기맞춤척도를계속고정. 검증/시험척도적합없음.",
        selection="모든학습전/후에같은gamma4개. 각규칙의학습후는같은규칙최적동결AP와같은gamma동결AP보다+.1%p. 자체gamma0대비AP+.1%p. 학습후gamma>0이어야함. 같은gamma동결F1손실<=.3%p·R100손실<=1%p·R1손실<=.3%p. BEV동일시간검증대비AP/F1열세2%p·R100/R1열세3%p이내. 통과중AP최대,작은gamma,작은lr,이른epoch. 통과후고정3시드확인,새시험없음.",
        limitations="반복검증·단일초기시드탐색. 학습승자고정. 추가학습전도학습된모델. 연산자고유우위아님. 새로운신뢰도가확률임을주장하지않음. 원고·그림유지.",
        sources={str(p.relative_to(ROOT)): run.HASH(p) for p in sources},
        test_used=False,
    )
    run.WRITE(path, spec)
    return spec


def activate() -> dict:
    """학습·검증 전용 경로를 기존 입력 엔진에 공급한다."""
    spec = prepare()
    run.OUT = OUT
    return spec


def model(checkpoint: Path = None) -> torch.nn.Module:
    """두 DiffAttnConv 층만 갱신 가능한 기존 모델을 불러온다."""
    spec = activate()
    net = prior.gate.MODEL(ROOT / spec["initial"] if checkpoint is None else checkpoint)
    for name, param in net.named_parameters():
        param.requires_grad_(name.startswith("gnn.convs."))
    return net


def evidence(banks: tuple, travel: np.ndarray, q: np.ndarray, j: np.ndarray) -> dict:
    """현재 시간 대응의 4채널 정렬 피크를 한 번 계산해 저장한다."""
    output = {k: [] for k in ("query", "candidate", "weight", "peaks")}
    n = len(travel)
    for start in range(0, len(q), 256):
        qq, jj = q[start : start + 256], j[start : start + 256]
        qn, jn, valid = neighbor_plans(travel, qq, jj)["arc_unique"]
        repeats = repeated_candidate_weights(jn, valid)
        weight = np.concatenate((np.ones(repeats.shape[:2] + (1,)), repeats), axis=-1)
        weight /= weight.sum(-1, keepdims=True)
        weight *= 11.0 / 15.0
        weight[..., 0] += 4.0 / 15.0
        shape = qn.shape[:2] + (1,)
        qi = np.concatenate((np.broadcast_to(qq[None, :, None], shape), qn), -1)
        ji = np.concatenate((np.broadcast_to(jj[None, :, None], shape), jn), -1)
        qi, ji = (
            np.clip(qi, 0, n - 1).transpose(1, 0, 2).copy(),
            np.clip(ji, 0, n - 1).transpose(1, 0, 2).copy(),
        )
        weight = weight.transpose(1, 0, 2).copy()
        ids, inverse = np.unique(qi * n + ji, return_inverse=True)
        channels = [
            correlation_profiles(b, ids // n, ids % n, ch)
            .max(1)[inverse]
            .reshape(qi.shape)
            for b in banks
            for ch in ("c_spec", "c_col")
        ]
        output["query"].append(qi)
        output["candidate"].append(ji)
        output["weight"].append(weight)
        output["peaks"].append(np.stack(channels, -1).astype(np.float64))
        assert np.all(qi + 10 <= qq[:, None, None] + 15)
        assert np.all(ji + 10 <= qq[:, None, None] + 15)
    return {k: np.concatenate(v) for k, v in output.items()}


def score_all(key: np.ndarray, data: dict, modes: tuple = MODES) -> dict:
    """같은 키·정렬 자료에서 모든 고정 신뢰도 비중을 계산한다."""
    embedding = torch.from_numpy(key)
    result = {f"{m}_{g:g}": [] for m in modes for g in GAMMAS}
    with torch.no_grad():
        for start in range(0, len(data["peaks"]), 256):
            sl = slice(start, start + 256)
            plan = {
                k: torch.from_numpy(data[k][sl])
                for k in ("query", "candidate", "weight")
            }
            peaks = torch.from_numpy(data["peaks"][sl])
            for mode in modes:
                for gamma in GAMMAS:
                    score, _ = reliability_score(embedding, plan, peaks, gamma, mode)
                    result[f"{mode}_{gamma:g}"].append(score.numpy())
    return {k: np.concatenate(v) for k, v in result.items()}


def make_pool(sequence: str) -> None:
    """초기 모델의 실제 학습 승자와 대응별 정렬 증거를 만든다."""
    spec = activate()
    folder = OUT / "pools" / sequence.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "audit.json"
    if target.exists():
        assert run.HASH(folder / "pool.npz") == run.READ(target)["sha256"]
        return
    tick = time.time()
    cache, pos, minimum, travel, bank, nb, nc, paths = run.inputs("train", sequence)
    net = model()
    key = run.source.make_keys(net, cache["descriptors"], cache["poses"])
    row, _ = run.retrieve(key, bank, pos, minimum)
    data = evidence((nb, nc), travel, row["query_idx"], row["j_star"])
    scores = score_all(key, data)
    check = run.current.adoption.scores(
        nb, nc, travel, row["query_idx"], row["j_star"]
    )["score"]
    error = float(np.max(np.abs(scores["attenuate_0"] - check)))
    np.testing.assert_allclose(scores["attenuate_0"], check, rtol=0, atol=3e-7)
    eligible = (row["min_past_gt_dist"] <= 5) | (row["min_past_gt_dist"] > 25)
    correct = (row["min_past_gt_dist"] <= 5) & (row["top1_geo_dist"] <= 5)
    np.savez_compressed(
        folder / "pool.npz",
        **row,
        **{f"plan_{k}": v for k, v in data.items()},
        **scores,
        eligible=eligible,
        correct=correct,
    )
    run.WRITE(
        target,
        dict(
            sequence=sequence,
            rows=len(correct),
            eligible=int(eligible.sum()),
            correct=int(correct[eligible].sum()),
            zero_reproduction_max_abs=error,
            sha256=run.HASH(folder / "pool.npz"),
            inputs={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
            seconds=time.time() - tick,
            test_used=False,
        ),
    )
    print("POOL", sequence, round(time.time() - tick, 1), flush=True)


def calibrate() -> dict:
    """학습8의 균등 정오 승자로 규칙별 초기 점수의 척도를 고정한다."""
    spec = activate()
    target = OUT / "calibration.json"
    if target.exists():
        return run.READ(target)
    records = {}
    for mode in MODES:
        xs, ys, ws = [], [], []
        for seq in spec["train"]:
            data = run.ARR(OUT / "pools" / seq.replace("/", "_") / "pool.npz")
            mask = data["eligible"]
            y = data["correct"][mask].astype(np.float64)
            weight = np.zeros(len(y), np.float64)
            for v in (0, 1):
                use = y == v
                if use.any():
                    weight[use] = 1.0 / use.sum()
            weight /= weight.sum()
            xs.append(data[f"{mode}_1"][mask, None])
            ys.append(y)
            ws.append(weight)
        records[mode] = run.fit_score(
            np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        )
    run.WRITE(target, records)
    return records


def train(arm: str) -> None:
    """신뢰도로 조정된 최종 판정 손실을 기존 두 층에 역전파한다."""
    spec, fit = activate(), calibrate()
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / "training_complete.json").exists():
        return
    assert not list(folder.glob("epoch*.pth")), "부분 학습을 덮어쓰지 않는다"
    mode, lr = ARMS[arm]
    torch.manual_seed(1)
    net = model()
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    parameters = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=lr, weight_decay=spec["weight_decay"])
    slope = float(fit[mode]["coefficients"][0] / fit[mode]["scale"][0])
    intercept = float(fit[mode]["intercept"] - slope * fit[mode]["mean"][0])
    records, hashes, updates = [], {}, 0
    tick = time.time()
    for epoch in range(1, max(EPOCHS) + 1):
        for ordinal, seq in enumerate(spec["train"]):
            name = seq.replace("/", "_")
            data = run.ARR(OUT / "pools" / name / "pool.npz")
            graph = torch.load(
                PARENT / "pools" / name / "graph.pt",
                map_location="cuda",
                weights_only=False,
            )
            tensor = {
                k: torch.from_numpy(v).cuda()
                for k, v in data.items()
                if k.startswith("plan_") or k == "correct"
            }
            positive = np.flatnonzero(data["eligible"] & data["correct"])
            negative = np.flatnonzero(data["eligible"] & ~data["correct"])
            rng = np.random.default_rng(39000 + 1000 * epoch + ordinal)
            losses, grads, orders, masses = [], [], [], []
            for step in range(spec["steps_per_sequence"]):
                sets = [v for v in (positive, negative) if len(v)]
                chosen = np.concatenate(
                    [
                        rng.choice(
                            v,
                            spec["batch_queries"] // len(sets),
                            replace=len(v) < spec["batch_queries"] // len(sets),
                        )
                        for v in sets
                    ]
                )
                chosen = rng.permutation(chosen)
                orders.append(run.array_hash(chosen))
                ix = torch.from_numpy(chosen).cuda()
                opt.zero_grad(set_to_none=True)
                key = F.normalize(net(graph), dim=-1)
                plan = {
                    k: tensor["plan_" + k][ix] for k in ("query", "candidate", "weight")
                }
                score, mass = reliability_score(
                    key, plan, tensor["plan_peaks"][ix], spec["training_gamma"], mode
                )
                loss = F.binary_cross_entropy_with_logits(
                    slope * score + intercept, tensor["correct"][ix].to(score.dtype)
                )
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(parameters, spec["clip"])
                assert torch.isfinite(grad)
                opt.step()
                updates += 1
                losses.append(float(loss))
                grads.append(float(grad))
                masses.append(float(mass.mean()))
            record = dict(
                epoch=epoch,
                sequence=seq,
                updates=updates,
                loss=float(np.mean(losses)),
                grad=float(np.mean(grads)),
                mass=float(np.mean(masses)),
                batch_hashes=orders,
            )
            records.append(record)
            run.WRITE(folder / "progress.json", record)
            del graph, key, tensor, loss, score, mass
            torch.cuda.empty_cache()
        if epoch in EPOCHS:
            checkpoint = folder / f"epoch{epoch}.pth"
            torch.save(
                dict(
                    model_state_dict=net.state_dict(),
                    epoch=epoch,
                    updates=updates,
                    protocol_sha256=run.HASH(OUT / "protocol.json"),
                ),
                checkpoint,
            )
            hashes[checkpoint.name] = run.HASH(checkpoint)
        print("TRAIN", arm, epoch, updates, flush=True)
    changed = []
    for name, value in net.state_dict().items():
        assert torch.isfinite(value).all(), name
        if not torch.equal(value.detach().cpu(), initial[name]):
            assert name.startswith("gnn.convs."), name
            changed.append(name)
    assert changed and updates == 512
    run.WRITE(
        folder / "training_complete.json",
        dict(
            status="passed",
            arm=arm,
            updates=updates,
            changed_parameters=changed,
            records=records,
            checkpoints=hashes,
            seconds=time.time() - tick,
            test_used=False,
        ),
    )


def validate(arm: str, epoch: int, sequence: str) -> None:
    """새 키로 전체 검색을 다시 하고 신뢰도를 최종 판정까지 평가한다."""
    spec = activate()
    folder = OUT / arm / f"epoch{epoch}/val"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (sequence.replace("/", "_") + ".json")
    if target.exists():
        assert run.READ(target)["status"] == "passed"
        return
    checkpoint = (
        ROOT / spec["initial"] if arm == "frozen" else OUT / arm / f"epoch{epoch}.pth"
    )
    tick = time.time()
    net = model(checkpoint)
    cache, pos, minimum, travel, bank, nb, nc, paths = run.inputs("val", sequence)
    key = run.source.make_keys(net, cache["descriptors"], cache["poses"])
    row, _ = run.retrieve(key, bank, pos, minimum)
    name = sequence.replace("/", "_")
    reference = run.ARR(prior.BASELINE / "val/s1" / (name + "_frozen_full.npz"))
    row["group"] = reference["group"]
    q, j = row["query_idx"], row["j_star"]
    np.testing.assert_array_equal(q, reference["query_idx"])
    np.testing.assert_array_equal(minimum[q], reference["min_past_gt_dist"])
    data = evidence((nb, nc), travel, q, j)
    modes = MODES if arm == "frozen" else (ARMS[arm][0],)
    scores = score_all(key, data, modes)
    unweighted = run.current.adoption.scores(nb, nc, travel, q, j)["score"]
    error = float(np.max(np.abs(scores[f"{modes[0]}_0"] - unweighted)))
    np.testing.assert_allclose(scores[f"{modes[0]}_0"], unweighted, rtol=0, atol=3e-7)
    metrics = {k: run.base.independent_metrics(row, v) for k, v in scores.items()}
    for k, v in scores.items():
        second = run.white.prior.BASE.exp.metrics(v, row)
        for m in run.base.METRICS:
            if metrics[k][m] is not None:
                np.testing.assert_allclose(metrics[k][m], second[m], rtol=0, atol=1e-12)
    # 같은 시작 모델 승자에 대한 연결 효과를 검색 변화와 분리한다.
    if arm == "frozen":
        np.savez_compressed(folder / (name + "_evidence.npz"), **data)
        fixed = metrics
    else:
        frozen_path = OUT / "frozen/epoch0/val" / (name + ".npz")
        frozen_row = run.ARR(frozen_path)
        fd = run.ARR(frozen_path.with_name(name + "_evidence.npz"))
        values = score_all(key, fd, modes)
        fixed = {
            k: run.base.independent_metrics(frozen_row, v) for k, v in values.items()
        }
    checks = []
    for at in (len(q) // 3, 2 * len(q) // 3):
        t, end = int(q[at]), int(q[at]) + 16
        small = run.source.make_keys(net, cache["descriptors"], cache["poses"], end)
        np.testing.assert_allclose(small[: t + 6], key[: t + 6], rtol=4e-5, atol=4e-6)
        trunc = run.current.adoption.previous.source.base.exp.truncate_bank
        weight = run.cal.fusion_weights(spec["confidence"])[
            run.cal.FUSIONS.index("aux_half")
        ]
        assert (
            prior.stages.query_search(
                small, trunc(bank, end), pos[:end], t, weight, {"full": (60, 200)}
            )[0]["full"]
            == j[at]
        )
        truncated = evidence(
            (trunc(nb, end), trunc(nc, end)),
            travel[:end],
            q[at : at + 1],
            j[at : at + 1],
        )
        np.testing.assert_allclose(
            truncated["peaks"], data["peaks"][at : at + 1], rtol=0, atol=0
        )
        np.testing.assert_array_equal(truncated["weight"], data["weight"][at : at + 1])
        values = score_all(small, truncated, modes)
        for k, v in values.items():
            np.testing.assert_allclose(v, scores[k][at : at + 1], atol=2e-6, rtol=0)
        checks.append(t)
    ledger = target.with_suffix(".npz")
    np.savez_compressed(ledger, **row, **scores)
    run.WRITE(
        target,
        dict(
            status="passed",
            arm=arm,
            epoch=epoch,
            sequence=sequence,
            metrics=metrics,
            fixed_winner_metrics=fixed,
            zero_reproduction_max_abs=error,
            prefix_checks=checks,
            ledger_sha256=run.HASH(ledger),
            checkpoint_sha256=run.HASH(checkpoint),
            input_sha256={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
            seconds=time.time() - tick,
            test_used=False,
        ),
    )
    print("VAL", arm, epoch, sequence, round(time.time() - tick, 1), flush=True)


def collect(arm: str, epoch: int, mode: str, gamma: float, spec: dict) -> dict:
    """최종 원장을 양성 시퀀스·계열 균등으로 집계한다."""
    rows, fixed = {}, {}
    for seq in spec["val"]:
        path = OUT / arm / f"epoch{epoch}/val" / (seq.replace("/", "_") + ".json")
        rec = run.READ(path)
        assert rec["status"] == "passed"
        assert run.HASH(path.with_suffix(".npz")) == rec["ledger_sha256"]
        rows[seq] = rec["metrics"][f"{mode}_{gamma:g}"]
        fixed[seq] = rec["fixed_winner_metrics"][f"{mode}_{gamma:g}"]
    return dict(
        arm=arm,
        epoch=epoch,
        mode=mode,
        gamma=gamma,
        **run.base.macro(rows),
        sequences=rows,
        fixed_winner=run.base.macro(fixed),
    )


def finish() -> None:
    """동결 최적·같은 설정·동일 시간 BEV 대조를 모두 확인한다."""
    spec = activate()
    frozen = {}
    results = {}
    eligible = []
    bev = run.READ(prior.gate.previous.OUT / "bev_validation.json")["temporal"]["macro"]
    for m in MODES:
        frozen[m] = {str(g): collect("frozen", 0, m, g, spec) for g in GAMMAS}
    best = {
        m: max(v.values(), key=lambda r: r["macro"]["AP"]) for m, v in frozen.items()
    }
    orders = []
    for arm, (mode, _) in ARMS.items():
        train = run.READ(OUT / arm / "training_complete.json")
        assert train["updates"] == 512
        orders.append([r["batch_hashes"] for r in train["records"]])
        for epoch in EPOCHS:
            zero = collect(arm, epoch, mode, 0.0, spec)
            for gamma in GAMMAS:
                r = collect(arm, epoch, mode, gamma, spec)
                base = frozen[mode][str(gamma)]["macro"]
                delta = {m: 100 * (r["macro"][m] - base[m]) for m in run.base.METRICS}
                checks = dict(
                    uses_reliability=gamma > 0,
                    AP_gain=delta["AP"] >= 0.1,
                    beats_best_frozen=r["macro"]["AP"]
                    >= best[mode]["macro"]["AP"] + 0.001,
                    beats_uniform=r["macro"]["AP"] >= zero["macro"]["AP"] + 0.001,
                    F1_preserved=delta["max_F1"] >= -0.3,
                    R100_preserved=delta["R@100P"] >= -1.0,
                    R1_preserved=delta["R@1"] >= -0.3,
                )
                checks.update(
                    {
                        f"bev_{m}": r["macro"][m] >= bev[m] - tol
                        for m, tol in {
                            "AP": 0.02,
                            "max_F1": 0.02,
                            "R@100P": 0.03,
                            "R@1": 0.03,
                        }.items()
                    }
                )
                r.update(delta_pp=delta, advancement_checks=checks)
                results[f"{arm}_{epoch}_{gamma:g}"] = r
                if all(checks.values()):
                    eligible.append(r)
    assert all(o == orders[0] for o in orders)
    selected = (
        min(
            eligible,
            key=lambda r: (
                -round(r["macro"]["AP"], 12),
                r["gamma"],
                ARMS[r["arm"]][1],
                r["epoch"],
            ),
        )
        if eligible
        else None
    )
    run.WRITE(
        OUT / "summary.json",
        dict(
            frozen=frozen,
            best_frozen=best,
            results=results,
            selected=selected,
            bev_temporal=bev,
            protocol_sha256=run.HASH(OUT / "protocol.json"),
            test_used=False,
        ),
    )
    run.WRITE(
        OUT / "completion_validation.json",
        dict(
            status="passed",
            ledgers=81,
            prefix_checks=162,
            updates_per_arm=512,
            equal_training_batches=True,
            test_used=False,
        ),
    )
    run.WRITE(
        OUT / "status.json",
        dict(status="validation_complete", selected=selected, test_used=False),
    )
    print(
        "FINISHED",
        (
            None
            if selected is None
            else {k: selected[k] for k in ("arm", "epoch", "gamma", "macro")}
        ),
        flush=True,
    )


def campaign(workers: int) -> None:
    """풀 생성·학습·실제 최종 평가를 모두 완료한다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    def call(args: list, name: str) -> None:
        with (logs / (name + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, __file__, *args],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def jobs(items: list, phase: str) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            fs = [pool.submit(call, args, name) for args, name in items]
            for done, f in enumerate(concurrent.futures.as_completed(fs), 1):
                f.result()
                run.WRITE(
                    OUT / "status.json",
                    dict(status="running", phase=phase, done=done, total=len(items)),
                )
                print("PROGRESS", phase, done, len(items), flush=True)

    jobs(
        [
            (["pool", "--sequence", s], "pool_" + s.replace("/", "_"))
            for s in spec["train"]
        ],
        "pools",
    )
    calibrate()
    for arm in ARMS:
        call(["train", "--arm", arm], "train_" + arm)
    jobs(
        [
            (
                ["val", "--arm", "frozen", "--epoch", "0", "--sequence", s],
                "val_frozen_" + s.replace("/", "_"),
            )
            for s in spec["val"]
        ],
        "frozen_validation",
    )
    jobs(
        [
            (
                ["val", "--arm", a, "--epoch", str(e), "--sequence", s],
                f'val_{a}_{e}_{s.replace("/","_")}',
            )
            for a in ARMS
            for e in EPOCHS
            for s in spec["val"]
        ],
        "learned_validation",
    )
    finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "action",
        choices=("prepare", "pool", "calibrate", "train", "val", "finish", "run"),
    )
    p.add_argument("--arm", default="attenuate_lr5")
    p.add_argument("--epoch", type=int, default=4)
    p.add_argument("--sequence", default="kitti/06")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    run.configure()
    if a.action == "prepare":
        prepare()
    elif a.action == "pool":
        make_pool(a.sequence)
    elif a.action == "calibrate":
        calibrate()
    elif a.action == "train":
        train(a.arm)
    elif a.action == "val":
        validate(a.arm, a.epoch, a.sequence)
    elif a.action == "finish":
        finish()
    else:
        campaign(a.workers)
