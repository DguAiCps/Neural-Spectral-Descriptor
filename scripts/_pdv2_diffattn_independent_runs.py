#!/usr/bin/env python3
"""DiffAttnConv 추가 학습을 표본 추출 시드만 바꿔 독립 반복하고, 시험46에서 학습 이득의 실행 간 변동을 잰다.

기존 학습(`_pdv2_diffattn_complex_learn.train`)과 같은 코드·입력·손실·512갱신을 쓴다. 바꾸는 것은 표본 추출 난수
흐름 하나다. 추가 학습은 eval 모드(드롭아웃 꺼짐)이고 새 매개변수의 무작위 초기화가 없으므로, 시작점이 같으면
학습 무작위성은 표본 추출뿐이다. 실행 시드 0은 기존 공식(45000+1000*epoch+순번)을 그대로 써서 기존 체크포인트를
비트 단위로 재현하는지 먼저 확인한다. 시험 평가는 기존 `_pdv2_contrast_pipeline_test` 실행기를 그대로 쓴다.

산출물은 C 드라이브 용량 때문에 D 드라이브(컨테이너 /workspace/data)에 둔다.
사용법(컨테이너 ring_baseline 안):
  python scripts/_pdv2_diffattn_independent_runs.py prepare      # 사전 등록·출처 고정(학습 전)
  python scripts/_pdv2_diffattn_independent_runs.py reproduce    # 실행 시드0으로 기존 6개 체크포인트 재현
  python scripts/_pdv2_diffattn_independent_runs.py train_all    # 3시작점×5실행×2조건
  python scripts/_pdv2_diffattn_independent_runs.py evaluate     # 시험46 평가
  python scripts/_pdv2_diffattn_independent_runs.py analyze      # 사전 등록한 분석
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

OUT = Path(
    os.environ.get(
        "NSD_INDEP_OUT", "/workspace/data/nsd_results/pdv2_diffattn_independent_20261108"
    )
)
EVAL = OUT / "test_eval"
STARTS = (1, 2, 3)
RUNS = (1, 2, 3, 4, 5)
ARMS = {"convs_only": 1e-4, "diff": 1e-4}
PRETRAIN = ROOT / "results/pdv2_spectral_keys_20261025"
TRAIN_INPUTS = ROOT / "results/pdv2_diffattn_contrast_learn_20261104/train_inputs"
RECAL = ROOT / "results/pdv2_diffattn_contrast_recalibrated_20261104"
OLD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
BOOTSTRAP = 10000
BOOT_SEED = 20261108

PREREGISTRATION = dict(
    purpose="DiffAttnConv 학습 이득(원고: 세 시작점에서 AP +0.13%p 등)이 추가 학습의 무작위성을 넘는지 확인한다. 기존 대조는 세 시작점의 추가 학습이 같은 표본 순서를 써서 학습 변동이 구간에 들어 있지 않았다.",
    fixed_settings="원고와 같은 설정을 바꾸지 않는다: 시작점 3개(사전학습 시드1~3의 20에포크), 학습8 시퀀스·정답/오답 균등 64표본·시퀀스당 8갱신×8에포크=512갱신, AdamW lr1e-4·감쇠1e-5·기울기 절단1, 재보정 손실 척도, matching_only 연결, 융합 가중치·수락 규칙. 시험셋으로 어떤 설정도 고르지 않는다.",
    randomness="실행 시드 r>=1은 표본 추출 난수를 numpy default_rng([r, 45000+1000*epoch+순번])로 바꾼다(추출되는 표본과 순서가 모두 달라짐). r=0은 기존 공식이며 재현 검사에만 쓴다. eval 모드라 드롭아웃이 없고 새 매개변수의 무작위 초기화가 없어 이것이 추가 학습의 유일한 무작위성이다.",
    design="시작점 3 × 실행 시드 5(r=1..5) × 학습 조건 2. 주 조건 convs_only(DiffAttnConv 두 층만 학습, 원고의 학습 기여 분리), 부 조건 diff(전체 학습, 원고 NSD). 대조는 결정적 고정 감산 fixed. 실행 횟수는 결과를 보기 전에 5로 고정하며 늘리거나 줄이지 않는다.",
    metrics="AP, 최대 F1, R@100P, R@1. 원고와 같은 집계: 시험46 중 양성이 있는 45시퀀스 → 도시6 균등 → 실행 평균.",
    estimand="각 학습 조건에서 (조건 − fixed) 차이의 평균. 시작점 3개는 고정한 모델로 보고 재표집하지 않는다(원고와 같음).",
    intervals=dict(
        A="학습 변동만: 실행별 매크로 차이 x_kr에서 시작점별 평균과 표준편차 s_k를 구하고, 평균=시작점 평균의 평균, SE=sqrt(Σ_k s_k²/5)/3, 자유도12 t 95% 구간. 시험 시퀀스는 고정.",
        B="학습 변동+시험 시퀀스: 10,000회. 도시 안에서 시퀀스 수를 유지해 복원추출하고, 시작점마다 실행 5개를 복원추출한 뒤 매크로 차이를 계산한 백분위 95% 구간. 난수 시드 20261108.",
        C="시퀀스만(원고 방식과 비교용): 실행을 고정하고 시퀀스만 B와 같이 재표집.",
    ),
    decision="주 조건 convs_only 기준. 유지: AP·최대F1·R@100P 세 지표 모두 구간B 하한>0. 부분 유지: AP 평균>0이나 세 지표 중 하나라도 구간B가 0을 포함. 유지 안 됨: AP 평균<=0. 실행 15개 중 차이>0인 개수도 함께 보고한다. diff는 같은 기준으로 부 결과로 보고한다.",
    reporting="모든 실행을 보고하고 결과를 보고 제외하지 않는다. 학습이 비유한 손실로 멈추면 그 실행을 실패로 보고한다. 재현 검사(r=0 체크포인트 6개 비트 일치, 시험 fixed·diff_s2 원장 일치)가 하나라도 실패하면 본 실행 전에 멈춘다. 원고·그림은 결과를 보고 사용자가 정하기 전까지 바꾸지 않는다.",
    exposure="시험셋은 과거 개발 노출이 있는 탐색적 평가이며 이번 실행은 설정 선택에 쓰지 않는다.",
)


def READ(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def WRITE(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def HASH(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def original_checkpoints() -> dict:
    """기존 시험 프로토콜에 기록된 r=0 체크포인트(해시 포함)."""
    cases = READ(OLD_TEST / "protocol.json")["cases"]
    out = {}
    for arm in ARMS:
        for k in STARTS:
            c = cases[f"{arm}_s{k}"]
            out[f"{arm}_s{k}"] = dict(checkpoint=c["checkpoint"], sha256=c["checkpoint_sha256"])
    return out


def prepare() -> dict:
    """사전 등록과 출처 해시를 학습 전에 고정한다. 이미 있으면 출처가 그대로인지 검사한다."""
    path = OUT / "protocol.json"
    if path.exists():
        spec = READ(path)
        for p, h in spec["sources"].items():
            assert HASH(ROOT / p) == h, p
        return spec
    old_test = READ(OLD_TEST / "protocol.json")
    recal = READ(RECAL / "protocol.json")
    sources = [Path(__file__), OLD_TEST / "protocol.json", OLD_TEST / "summary.json",
               RECAL / "protocol.json"]
    sources += [PRETRAIN / f"s{k}/octave288/checkpoints/checkpoint_epoch_20.pth" for k in STARTS]
    sources += sorted(TRAIN_INPUTS.glob("*.pt"))
    for c in original_checkpoints().values():
        p = ROOT / c["checkpoint"]
        assert HASH(p) == c["sha256"], p
        sources.append(p)
    spec = dict(
        created_unix=time.time(),
        preregistration=PREREGISTRATION,
        starts=list(STARTS),
        runs=list(RUNS),
        arms=ARMS,
        train=recal["train"],
        loss_fit=recal["loss_fit"],
        test=old_test["test"],
        original=original_checkpoints(),
        output_root=str(OUT),
        sources={rel(p): HASH(p) for p in sources},
    )
    for name in spec["train"]:
        graph = TRAIN_INPUTS / (name.replace("/", "_") + ".pt")
        assert HASH(graph) == READ(graph.with_suffix(".json"))["graph_sha256"], graph
    WRITE(path, spec)
    WRITE(OUT / "preregistration.json", dict(created_unix=spec["created_unix"], **PREREGISTRATION))
    print("PREPARED", path, flush=True)
    return spec


# ---------------------------------------------------------------- 학습
def trainer():
    """기존 학습 사슬을 불러오고 손실 척도를 이 실행의 protocol.json에서 읽게 한다."""
    import torch
    import _pdv2_diffattn_contrast_recalibrated as R

    e = R.e
    R.OUT = OUT  # R.score는 학습(미분) 중에만 OUT/protocol.json의 loss_fit을 쓴다
    e.configure()
    return torch, R, e


def build(torch, e, start: int, arm: str):
    """시작점의 사전학습 가중치에서 기존과 같은 구조·초기화(균등 감산 -0.5)를 만든다."""
    from gnn.contrast_spectral_difference import ContrastSpectralDifference

    torch.manual_seed(start)  # 기존 실행과 같음. 무작위 초기화 매개변수가 없어 결과와 무관
    ref = e.run.source.factory(288)
    path = PRETRAIN / f"s{start}/octave288/checkpoints/checkpoint_epoch_20.pth"
    ref.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"], strict=True
    )
    net = ContrastSpectralDifference(ref, "diff").cuda().eval()
    if arm == "convs_only":
        for name, p in net.named_parameters():
            p.requires_grad_(name.startswith("convs."))
    return net


def train(start: int, run_seed: int, arm: str, folder: Path) -> Path:
    """기존 train()과 같은 512갱신. 표본 추출 난수만 run_seed로 바꾼다."""
    import torch.nn.functional as F

    torch, R, e = trainer()
    spec = prepare()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "epoch8.pth"
    if (folder / "training_complete.json").exists():
        return target
    assert not target.exists()
    net = build(torch, e, start, arm)
    graph0 = torch.load(TRAIN_INPUTS / (spec["train"][0].replace("/", "_") + ".pt"),
                        map_location="cuda", weights_only=False)
    with torch.no_grad():
        init_error = float((net(graph0) - e.prior.filtered(graph0, -0.5)).abs().max())
    assert init_error < 4e-6, init_error
    del graph0
    initial = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    params = [v for v in net.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(params, lr=ARMS[arm], weight_decay=1e-5)
    records, updates, tick = [], 0, time.time()
    for epoch in range(1, 9):
        for ordinal, s in enumerate(spec["train"]):
            name = s.replace("/", "_")
            graph = torch.load(TRAIN_INPUTS / (name + ".pt"), map_location="cuda", weights_only=False)
            data = e.run.ARR(e.prior.PARENT / "pools/train" / name / "pool.npz")
            plan = {k: torch.from_numpy(data["plan_" + k]).cuda() for k in ("query", "candidate", "weight")}
            sets = [np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)]
            sets = [v for v in sets if len(v)]
            base = 45000 + 1000 * epoch + ordinal
            rng = np.random.default_rng(base if run_seed == 0 else [run_seed, base])
            orders, losses, grads = [], [], []
            for _ in range(8):
                n = 64 // len(sets)
                chosen = rng.permutation(
                    np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets])
                )
                orders.append(e.run.array_hash(chosen))
                ix = torch.from_numpy(chosen).cuda()
                opt.zero_grad(set_to_none=True)
                z = net(graph)
                value = e.prior.score(z, {k: v[ix] for k, v in plan.items()})
                loss = F.binary_cross_entropy_with_logits(
                    value, torch.from_numpy(data["correct"][chosen]).cuda().float()
                )
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
                assert torch.isfinite(grad)
                opt.step()
                updates += 1
                losses.append(float(loss.detach()))
                grads.append(float(grad))
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)),
                                grad=float(np.mean(grads)), batch_hashes=orders,
                                strength_mean=float(net.last_strengths.detach().mean())))
            del graph, z, value, loss, plan
    torch.save(dict(model_state_dict=net.state_dict(), epoch=8, updates=updates, kind=arm,
                    start=start, run_seed=run_seed, protocol_sha256=HASH(OUT / "protocol.json")),
               target)
    changed = [n for n, v in net.state_dict().items() if not torch.equal(v.cpu(), initial[n])]
    allowed = {n for n, p in net.named_parameters() if p.requires_grad}
    assert set(changed) <= allowed
    WRITE(folder / "training_complete.json",
          dict(status="passed", start=start, run_seed=run_seed, arm=arm, updates=updates,
               init_error=init_error, seconds=time.time() - tick, changed_parameters=changed,
               trainable_parameters=sum(p.numel() for p in params), records=records,
               sha256=HASH(target)))
    print("TRAINED", start, run_seed, arm, updates, round(time.time() - tick, 1), flush=True)
    return target


def reproduce() -> None:
    """실행 시드0이 기존 6개 체크포인트의 모든 텐서를 비트 단위로 재현하는지 확인한다."""
    import torch

    spec = prepare()
    rows = {}
    for arm in ARMS:
        for k in STARTS:
            new = train(k, 0, arm, OUT / "repro" / f"s{k}" / arm)
            old = ROOT / spec["original"][f"{arm}_s{k}"]["checkpoint"]
            a = torch.load(new, map_location="cpu", weights_only=False)["model_state_dict"]
            b = torch.load(old, map_location="cpu", weights_only=False)["model_state_dict"]
            assert a.keys() == b.keys()
            diff = {n: float((a[n].double() - b[n].double()).abs().max()) for n in a}
            exact = all(torch.equal(a[n], b[n]) for n in a)
            rows[f"{arm}_s{k}"] = dict(exact=exact, max_abs=max(diff.values()), original=rel(old))
            print("REPRO", arm, k, exact, max(diff.values()), flush=True)
    status = "passed" if all(r["exact"] for r in rows.values()) else "failed"
    WRITE(OUT / "reproduction.json", dict(status=status, checkpoints=rows))
    assert status == "passed", rows


def train_all() -> None:
    """3시작점×5실행×2조건을 GPU에서 세 개씩 학습한다."""
    assert READ(OUT / "reproduction.json")["status"] == "passed"
    jobs = [(k, r, arm) for arm in ARMS for k in STARTS for r in RUNS]
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(job):
        k, r, arm = job
        with (logs / f"train_s{k}_r{r}_{arm}.log").open("a") as stream:
            subprocess.run([sys.executable, __file__, "train", "--start", str(k), "--run", str(r),
                            "--arm", arm], stdout=stream, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for i, f in enumerate(concurrent.futures.as_completed([pool.submit(call, j) for j in jobs]), 1):
            f.result()
            print("TRAIN_DONE", i, len(jobs), flush=True)


def run_folder(k: int, r: int, arm: str) -> Path:
    return OUT / "runs" / f"s{k}" / f"r{r}" / arm


# ---------------------------------------------------------------- 시험 평가
def eval_cases() -> dict:
    """새 30개와 재현 확인용 fixed·diff_s2_r0을 고정한다."""
    cases = {
        "fixed": dict(kind="fixed", seed=0, checkpoint=None),
        "diff_s2_r0": dict(kind="diff", seed=2,
                           checkpoint=str(OUT / "repro/s2/diff/epoch8.pth")),
    }
    for arm in ARMS:
        for k in STARTS:
            for r in RUNS:
                cases[f"{arm}_s{k}_r{r}"] = dict(kind=arm, seed=k,
                                                 checkpoint=str(run_folder(k, r, arm) / "epoch8.pth"))
    for c in cases.values():
        if c["checkpoint"]:
            path = Path(c["checkpoint"])
            c["checkpoint_sha256"] = HASH(path)
            assert READ(path.parent / "training_complete.json")["sha256"] == c["checkpoint_sha256"]
    return cases


def eval_prepare() -> dict:
    """기존 시험 명세에서 사례만 바꾼 평가 명세를 고정한다."""
    path = EVAL / "protocol.json"
    if path.exists():
        return READ(path)
    spec = dict(READ(OLD_TEST / "protocol.json"))
    spec.update(created_unix=time.time(), cases=eval_cases(),
                parent_protocol_sha256=HASH(OUT / "protocol.json"),
                test_policy="사전 등록한 독립 반복 학습의 고정 시험 평가. 설정 선택 없음.")
    WRITE(path, spec)
    return spec


def eval_worker(sequence: str) -> None:
    """기존 시험 실행기로 한 시퀀스의 모든 사례를 평가하고 재현 원장을 대조한다."""
    import _pdv2_contrast_pipeline_test as T

    T.OUT = EVAL
    T.prepare = eval_prepare
    T.pipeline.e.configure()
    name = sequence.replace("/", "_")
    folder = EVAL / "test" / name
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        src = OLD_TEST / "test" / name / ("candidate_pool" + suffix)
        dst = folder / ("candidate_pool" + suffix)
        if not dst.exists():
            shutil.copy2(src, dst)
    T.worker("test", sequence)
    checks = {}
    for tag, old in (("fixed", "fixed"), ("diff_s2_r0", "diff_s2")):
        new = np.load(folder / f"{tag}.npz")
        ref = np.load(OLD_TEST / "test" / name / f"{old}.npz")
        for field in ("query_idx", "j_star"):
            np.testing.assert_array_equal(new[field], ref[field])
        error = float(np.abs(new["score"] - ref["score"]).max())
        assert error < 8e-6, (tag, error)
        checks[tag] = dict(winners_exact=True, score_error=error)
    WRITE(folder / "reproduction_check.json", dict(status="passed", checks=checks))


def evaluate() -> None:
    """시험46을 세 시퀀스씩 평가한다."""
    assert READ(OUT / "reproduction.json")["status"] == "passed"
    spec = eval_prepare()
    logs = EVAL / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(sequence):
        with (logs / (sequence.replace("/", "_") + ".log")).open("a") as stream:
            subprocess.run([sys.executable, __file__, "eval_worker", "--sequence", sequence],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in spec["test"]]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            WRITE(EVAL / "status.json", dict(status="running", done=i, total=len(futures)))
            print("EVAL_DONE", i, len(futures), flush=True)
    WRITE(EVAL / "status.json", dict(status="complete"))


# ---------------------------------------------------------------- 분석
def analyze() -> None:
    """사전 등록한 추정량·세 구간·판정 규칙을 계산한다."""
    from scipy import stats as st
    import _pdv2_pilot as pilot

    spec = READ(OUT / "protocol.json")
    ev = READ(EVAL / "protocol.json")
    metrics = ("AP", "max_F1", "R@100P", "R@1")
    per = {}
    for s in ev["test"]:
        folder = EVAL / "test" / s.replace("/", "_")
        assert READ(folder / "reproduction_check.json")["status"] == "passed"
        per[s] = READ(folder / "summary.json")["metrics"]
    cities = {c: [s for s in seqs if per[s]["fixed"]["n_pos"]] for c, seqs in pilot.CITY.items()}
    assert sum(len(v) for v in cities.values()) == 45

    def macro(values: dict) -> float:
        return float(np.mean([np.mean([values[s] for s in seqs]) for seqs in cities.values()]))

    old = READ(OLD_TEST / "summary.json")
    fixed_macro = {m: macro({s: per[s]["fixed"][m] for s in per}) for m in metrics}
    for m in metrics:  # 기존 보고값 재현
        assert abs(fixed_macro[m] - old["mean"]["fixed"][m]) < 1e-12, m

    rng = np.random.default_rng(BOOT_SEED)
    order = [s for seqs in cities.values() for s in seqs]
    city_idx = [np.array([order.index(s) for s in seqs]) for seqs in cities.values()]
    boot_seq = [[rng.integers(0, len(ix), len(ix)) for ix in city_idx] for _ in range(BOOTSTRAP)]
    boot_run = rng.integers(0, len(RUNS), (BOOTSTRAP, len(STARTS), len(RUNS)))

    def macro_arr(vec: np.ndarray, picks=None) -> float:
        parts = []
        for c, ix in enumerate(city_idx):
            sel = ix if picks is None else ix[picks[c]]
            parts.append(vec[sel].mean())
        return float(np.mean(parts))

    report = dict(fixed_macro_percent={m: 100 * v for m, v in fixed_macro.items()})
    tcrit = float(st.t.ppf(0.975, len(STARTS) * (len(RUNS) - 1)))
    for arm in ARMS:
        res = {}
        for m in metrics:
            # D[k, r, seq] (퍼센트포인트)
            D = np.array([[[100 * (per[s][f"{arm}_s{k}_r{r}"][m] - per[s]["fixed"][m]) for s in order]
                           for r in RUNS] for k in STARTS])
            runs = np.array([[macro_arr(D[i, j]) for j in range(len(RUNS))] for i in range(len(STARTS))])
            mean = float(runs.mean())
            se = float(np.sqrt((runs.std(axis=1, ddof=1) ** 2 / len(RUNS)).sum()) / len(STARTS))
            mean_seq = D.mean(axis=(0, 1))
            boot_b, boot_c = np.empty(BOOTSTRAP), np.empty(BOOTSTRAP)
            for b in range(BOOTSTRAP):
                pick = D[np.arange(len(STARTS))[:, None], boot_run[b]].mean(axis=(0, 1))
                boot_b[b] = macro_arr(pick, boot_seq[b])
                boot_c[b] = macro_arr(mean_seq, boot_seq[b])
            original = float(np.mean([100 * (old["per_case"][f"{arm}_s{k}"]["macro"][m]
                                              - old["mean"]["fixed"][m]) for k in STARTS]))
            res[m] = dict(
                mean_pp=mean,
                interval_A_runs_t=[mean - tcrit * se, mean + tcrit * se],
                interval_B_runs_and_sequences=[float(np.percentile(boot_b, 2.5)),
                                               float(np.percentile(boot_b, 97.5))],
                interval_C_sequences_only=[float(np.percentile(boot_c, 2.5)),
                                           float(np.percentile(boot_c, 97.5))],
                positive_runs=int((runs > 0).sum()),
                total_runs=int(runs.size),
                per_start_mean=[float(v) for v in runs.mean(axis=1)],
                per_start_sd=[float(v) for v in runs.std(axis=1, ddof=1)],
                per_run=runs.round(6).tolist(),
                original_r0_mean_pp=original,
            )
        lows = [res[m]["interval_B_runs_and_sequences"][0] for m in ("AP", "max_F1", "R@100P")]
        if res["AP"]["mean_pp"] <= 0:
            verdict = "유지 안 됨"
        elif all(v > 0 for v in lows):
            verdict = "유지"
        else:
            verdict = "부분 유지"
        report[arm] = dict(metrics=res, verdict=verdict)
    report.update(primary="convs_only", t_critical=tcrit, bootstrap=BOOTSTRAP, bootstrap_seed=BOOT_SEED,
                  aggregation="45시퀀스→도시6 균등→실행 평균(시작점 고정)",
                  test_used_for_selection=False, historical_test_exposure=True)
    WRITE(OUT / "analysis.json", report)
    for arm in ARMS:
        print(arm, report[arm]["verdict"], flush=True)
        for m, v in report[arm]["metrics"].items():
            print(f"  {m}: {v['mean_pp']:+.3f}  A{np.round(v['interval_A_runs_t'], 3)}"
                  f"  B{np.round(v['interval_B_runs_and_sequences'], 3)}"
                  f"  C{np.round(v['interval_C_sequences_only'], 3)}"
                  f"  양수 {v['positive_runs']}/{v['total_runs']}  기존r0 {v['original_r0_mean_pp']:+.3f}",
                  flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "reproduce", "train", "train_all", "evaluate",
                                      "eval_worker", "analyze"))
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--arm", default="convs_only", choices=tuple(ARMS))
    p.add_argument("--sequence", default="kitti/00")
    a = p.parse_args()
    if a.action == "train":
        assert a.start in STARTS and a.run in RUNS
        train(a.start, a.run, a.arm, run_folder(a.start, a.run, a.arm))
    elif a.action == "eval_worker":
        eval_worker(a.sequence)
    else:
        globals()[a.action]()
