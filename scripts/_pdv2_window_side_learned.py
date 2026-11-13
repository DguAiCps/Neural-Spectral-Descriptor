#!/usr/bin/env python3
"""시간 창 방향 대조의 학습판: DiffAttnConv를 과거 5개 이웃만으로 추가 학습하고 시험46에서 방향별로 잰다.

학습 없는 대조(`_pdv2_window_side_control.py`)에서 과거 전용 균일 감산(-0.5, `past_same`)은 역방향 재방문에서 크게
나빠졌다. 여기서는 같은 한쪽 창으로 DiffAttnConv를 학습해도 그 손실이 남는지 잰다.

과거 전용 창은 그래프 사본에서 시간 간선을 과거 이웃(src<tgt, 칸 0~4)만 남기고 전달 칸 6~10을 끈 것이다. 간선까지
줄이므로 두 층의 메시지 전달·주의·게이트도 과거만 본다. 초기 주의는 남은 간선 위의 균등이므로 초기 출력이 과거 전용
균일 감산 -0.5와 같다. 전달 칸만 끄면 주의가 미래 간선까지 나눠 가져 초기 세기가 절반이 되므로 쓰지 않는다.
학습(`_pdv2_diffattn_independent_runs.train`의 실행 시드 0과 같은 512갱신)과 시험 평가(`_pdv2_contrast_pipeline_test`)
모두 이 사본을 쓰고, 최종 시간 판정은 앞뒤 이웃을 그대로 쓴다. 대칭 창도 같은 코드로 다시 학습해 기존 체크포인트와
비트 단위로 같은지 확인한다. 시험셋으로 고르는 설정은 없다.

산출물은 D 드라이브(컨테이너 /workspace/data)에 둔다.
사용법(컨테이너 ring_baseline 안, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python scripts/_pdv2_window_side_learned.py train_all   # 창 2 x 조건 2 x 시작점 3 학습, 대칭 체크포인트 재현 확인
  python scripts/_pdv2_window_side_learned.py test        # 시험46 평가·기존 원장 재현 확인·방향별 분석
"""
import argparse
import concurrent.futures
import hashlib
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

OUT = Path(os.environ.get("NSD_SIDE_LEARNED_OUT",
                          "/workspace/data/nsd_results/pdv2_window_side_learned_20261111"))
TEST = OUT / "test_eval"
PRETRAIN = ROOT / "results/pdv2_spectral_keys_20261025"
TRAIN_INPUTS = ROOT / "results/pdv2_diffattn_contrast_learn_20261104/train_inputs"
RECAL = ROOT / "results/pdv2_diffattn_contrast_recalibrated_20261104"
OLD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
DIRECTIONS = ROOT / "results/pdv2_review_direction_yaw_20261105"
SIDE = Path("/workspace/data/nsd_results/pdv2_window_side_20261111")
WINDOWS = ("symmetric", "past")
ARMS = ("convs_only", "diff")
STARTS = (1, 2, 3)
METRICS = ("AP", "max_F1", "R@100P", "R@1")
GROUPS = ("all", "forward_only", "reverse_only")
BOOTSTRAP = 10000
BOOT_SEED = 20261111


def READ(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def WRITE(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def HASH(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tag(window: str, arm: str, start: int) -> str:
    """대칭 창은 기존 시험 원장과 같은 이름을 쓴다."""
    return f"{arm}_s{start}" if window == "symmetric" else f"past_{arm}_s{start}"


def run_folder(window: str, start: int, arm: str) -> Path:
    return OUT / "runs" / window / f"s{start}" / arm


def past_graph(graph):
    """시간 간선을 과거 이웃만 남기고 전달 칸 6~10을 끈 사본을 만든다. 칸 0~4가 과거, 6~10이 미래다."""
    import torch

    g = graph.clone()
    keep = g.edge_index[0] < g.edge_index[1]
    g.edge_index, g.edge_attr, g.edge_type = g.edge_index[:, keep], g.edge_attr[keep], g.edge_type[keep]
    g.transport_mask[:, 6:] = 0
    rebuilt = torch.zeros_like(g.transport_mask)
    src, tgt = g.edge_index
    rebuilt[tgt, src - tgt + 5] = 1
    assert torch.equal(rebuilt, g.transport_mask)  # 남은 간선과 전달 칸이 정확히 같다
    return g


def causal_check(net, graph) -> dict:
    """가운데 노드 출력의 입력 기울기가 닿는 노드 범위를 잰다. 과거 전용이면 미래 노드의 기울기가 정확히 0이다."""
    import torch

    g = graph.clone()
    g.x.requires_grad_(True)
    g.spectrum.requires_grad_(True)
    i = len(g.x) // 2
    gx, gs = torch.autograd.grad(net(g)[i].abs().sum(), [g.x, g.spectrum])
    size = (gx.abs().sum(1) + gs.abs().flatten(1).sum(1)).cpu().numpy()
    hit = np.flatnonzero(size > 0)
    return dict(node=i, future_max=float(size[i + 1:].max()), past_max=float(size[:i].max()),
                future_nodes=int((size[i + 1:] > 0).sum()), past_nodes=int((size[:i] > 0).sum()),
                reach=[int(hit.min() - i), int(hit.max() - i)])


# ---------------------------------------------------------------- 학습
def build(torch, e, start: int, arm: str):
    """`_pdv2_diffattn_independent_runs.build`와 같다: 사전학습 가중치에서 균등 감산 -0.5 초기화."""
    from gnn.contrast_spectral_difference import ContrastSpectralDifference

    torch.manual_seed(start)
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


def train(window: str, start: int, arm: str) -> Path:
    """`_pdv2_diffattn_independent_runs.train`의 실행 시드 0과 같은 512갱신. 과거 창이면 모든 학습 그래프를 사본으로 바꾼다."""
    import torch
    import torch.nn.functional as F
    import _pdv2_diffattn_contrast_recalibrated as R  # R.score는 학습(미분) 중 RECAL/protocol.json의 loss_fit을 쓴다

    e = R.e
    e.configure()
    assert R.OUT == RECAL
    sequences = READ(RECAL / "protocol.json")["train"]
    folder = run_folder(window, start, arm)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "epoch8.pth"
    if (folder / "training_complete.json").exists():
        return target
    assert not target.exists()

    def load(s: str):
        graph = torch.load(TRAIN_INPUTS / (s.replace("/", "_") + ".pt"), map_location="cuda",
                           weights_only=False)
        return past_graph(graph) if window == "past" else graph

    net = build(torch, e, start, arm)
    graph0 = load(sequences[0])
    with torch.no_grad():
        init_error = float((net(graph0) - e.prior.filtered(graph0, -0.5)).abs().max())
    assert init_error < 4e-6, init_error
    causal = dict(initial=causal_check(net, graph0))
    initial = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    params = [v for v in net.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
    records, updates, tick = [], 0, time.time()
    for epoch in range(1, 9):
        for ordinal, s in enumerate(sequences):
            graph = load(s)
            data = e.run.ARR(e.prior.PARENT / "pools/train" / s.replace("/", "_") / "pool.npz")
            plan = {k: torch.from_numpy(data["plan_" + k]).cuda() for k in ("query", "candidate", "weight")}
            sets = [np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng(45000 + 1000 * epoch + ordinal)  # 실행 시드 0(원고 표본 순서)
            losses = []
            for _ in range(8):
                n = 64 // len(sets)
                chosen = rng.permutation(
                    np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets])
                )
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
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)),
                                strength_mean=float(net.last_strengths.detach().mean())))
            del graph, z, value, loss, plan
    causal["final"] = causal_check(net, graph0)
    for c in causal.values():
        if window == "past":
            assert c["future_max"] == 0 and c["future_nodes"] == 0 and c["past_nodes"] > 0, c
        else:
            assert c["future_nodes"] > 0 and c["past_nodes"] > 0, c
    torch.save(dict(model_state_dict=net.state_dict(), epoch=8, updates=updates, kind=arm,
                    start=start, run_seed=0, window=window), target)
    changed = [n for n, v in net.state_dict().items() if not torch.equal(v.cpu(), initial[n])]
    assert set(changed) <= {n for n, p in net.named_parameters() if p.requires_grad}
    WRITE(folder / "training_complete.json",
          dict(status="passed", window=window, start=start, arm=arm, run_seed=0, updates=updates,
               init_error=init_error, causal_check=causal, seconds=time.time() - tick,
               trainable_parameters=sum(p.numel() for p in params), records=records,
               sha256=HASH(target)))
    print("TRAINED", window, start, arm, updates, round(time.time() - tick, 1), causal, flush=True)
    return target


def train_all() -> None:
    """창 2 x 조건 2 x 시작점 3을 차례로 학습하고 대칭 창이 기존 6개 체크포인트를 비트 단위로 재현하는지 본다."""
    import torch

    if not (OUT / "protocol.json").exists():
        WRITE(OUT / "protocol.json", dict(
            created_unix=time.time(),
            purpose="과거 5개 이웃만으로 DiffAttnConv를 학습해도 과거 전용 창의 역방향 손실이 남는지 확인한다.",
            window="과거 전용: 시간 간선 src<tgt만 남기고 전달 칸 6~10을 끈 그래프 사본을 학습·평가 모두에 쓴다. 최종 시간 판정은 앞뒤 그대로.",
            fixed_settings="원고와 같음: 사전학습 시드1~3 20에포크 시작점, 학습8, 512갱신, AdamW lr1e-4·감쇠1e-5·절단1, 재보정 손실, 실행 시드0 표본 순서, matching_only, 융합·수락 규칙. 시험셋으로 고르는 설정 없음.",
            train=READ(RECAL / "protocol.json")["train"],
            test=READ(OLD_TEST / "protocol.json")["test"]))
    for window in WINDOWS:
        for arm in ARMS:
            for k in STARTS:
                train(window, k, arm)
    cases = READ(OLD_TEST / "protocol.json")["cases"]
    rows = {}
    for arm in ARMS:
        for k in STARTS:
            old = ROOT / cases[f"{arm}_s{k}"]["checkpoint"]
            assert HASH(old) == cases[f"{arm}_s{k}"]["checkpoint_sha256"]
            a = torch.load(run_folder("symmetric", k, arm) / "epoch8.pth", map_location="cpu",
                           weights_only=False)["model_state_dict"]
            b = torch.load(old, map_location="cpu", weights_only=False)["model_state_dict"]
            assert a.keys() == b.keys()
            rows[f"{arm}_s{k}"] = dict(
                exact=all(torch.equal(a[n], b[n]) for n in a),
                max_abs=max(float((a[n].double() - b[n].double()).abs().max()) for n in a),
                original=cases[f"{arm}_s{k}"]["checkpoint"])
    status = "passed" if all(r["exact"] for r in rows.values()) else "failed"
    WRITE(OUT / "reproduction.json", dict(status=status, checkpoints=rows))
    print("REPRODUCTION", status, rows, flush=True)
    assert status == "passed", rows


# ---------------------------------------------------------------- 시험
def test_spec() -> dict:
    """기존 시험 명세에서 사례만 바꾼다. 정제 없음·대칭/과거 균일 감산·학습 12개."""
    path = TEST / "protocol.json"
    if path.exists():
        return READ(path)
    cases = {"identity": dict(kind="identity", seed=0, checkpoint=None),
             "fixed": dict(kind="fixed", seed=0, checkpoint=None),
             "past_fixed": dict(kind="past_fixed", seed=0, checkpoint=None)}
    for window in WINDOWS:
        for arm in ARMS:
            for k in STARTS:
                ckpt = run_folder(window, k, arm) / "epoch8.pth"
                digest = HASH(ckpt)
                assert READ(ckpt.parent / "training_complete.json")["sha256"] == digest
                cases[tag(window, arm, k)] = dict(kind=("past_" if window == "past" else "") + arm,
                                                  seed=k, checkpoint=str(ckpt), checkpoint_sha256=digest)
    spec = dict(READ(OLD_TEST / "protocol.json"))
    spec.update(created_unix=time.time(), cases=cases,
                test_policy="고정 설정의 시험 보고. 설정 선택 없음.")
    WRITE(path, spec)
    return spec


def patch(pipeline) -> None:
    """kind가 past_로 시작하면 과거 전용 그래프 사본에 원래 변환·모델을 그대로 쓴다."""
    transform, load_model = pipeline.transform, pipeline.load_model

    def past_transform(net, graph, kind: str):
        if kind.startswith("past_"):
            return transform(net, past_graph(graph), kind[len("past_"):])
        return transform(net, graph, kind)

    def past_load(case: dict):
        if case["kind"].startswith("past_"):
            case = dict(case, kind=case["kind"][len("past_"):])
        return load_model(case)

    pipeline.transform, pipeline.load_model = past_transform, past_load


def test_worker(sequence: str) -> None:
    """한 시퀀스의 15사례를 평가하고 대칭 사례는 기존 원장, past_fixed는 학습 없는 대조 원장과 대조한다."""
    import _pdv2_contrast_pipeline_test as T

    T.OUT = TEST
    T.prepare = test_spec
    T.pipeline.e.configure()
    patch(T.pipeline)
    name = sequence.replace("/", "_")
    folder = TEST / "test" / name
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        dst = folder / ("candidate_pool" + suffix)
        if not dst.exists():
            shutil.copy2(OLD_TEST / "test" / name / ("candidate_pool" + suffix), dst)
    T.worker("test", sequence)
    refs = {t: OLD_TEST / "test" / name / f"{t}.npz"
            for t in ["identity", "fixed"] + [tag("symmetric", a, k) for a in ARMS for k in STARTS]}
    refs["past_fixed"] = SIDE / "test_eval/test" / name / "past_0p5.npz"
    checks = {}
    for t, ref in refs.items():
        new, old = np.load(folder / f"{t}.npz"), np.load(ref)
        for field in ("query_idx", "j_star"):
            np.testing.assert_array_equal(new[field], old[field])
        error = float(np.abs(new["score"] - old["score"]).max())
        assert error < 8e-6, (t, error)
        checks[t] = dict(winners_exact=True, score_error=error)
    WRITE(folder / "reproduction_check.json", dict(status="passed", checks=checks))


def test() -> None:
    assert READ(OUT / "reproduction.json")["status"] == "passed"
    spec = test_spec()
    logs = TEST / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(sequence):
        with (logs / (sequence.replace("/", "_") + ".log")).open("a") as stream:
            subprocess.run([sys.executable, __file__, "test_worker", "--sequence", sequence],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in spec["test"]]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            print("TEST_WORKER DONE", i, len(futures), flush=True)
    analyze()


# ---------------------------------------------------------------- 분석
def analyze() -> None:
    """방향별 6도시 매크로, 시작점 평균·표준편차, 도시 안 시퀀스 짝지은 재표집 차이를 계산한다."""
    import _pdv2_pilot as pilot
    import _review_direction_metrics as direction

    spec = test_spec()
    rows = {}  # 시퀀스 -> 사례 -> 방향 -> 지표
    errors = []
    for s in spec["test"]:
        name = s.replace("/", "_")
        check = READ(TEST / "test" / name / "reproduction_check.json")
        assert check["status"] == "passed"
        errors += [v["score_error"] for v in check["checks"].values()]
        with np.load(DIRECTIONS / "directions" / (name + ".npz")) as d:
            label = {k: d[k] for k in ("query_idx", "group")}
        masks = dict(all=None, forward_only=label["group"] == 0, reverse_only=label["group"] == 1)
        rows[s] = {}
        for t in spec["cases"]:
            with np.load(TEST / "test" / name / (t + ".npz")) as z:
                np.testing.assert_array_equal(z["query_idx"], label["query_idx"])
                rows[s][t] = {g: direction.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"], m)
                              for g, m in masks.items()}

    # 양성이 있는 시퀀스는 사례와 무관하다. 도시별 시퀀스 값 배열을 만든다.
    valid = {g: {c: [s for s in seqs if rows[s]["fixed"][g] is not None] for c, seqs in pilot.CITY.items()}
             for g in GROUPS}
    for s in rows:
        for g in GROUPS:
            assert len({rows[s][t][g] is None for t in spec["cases"]}) == 1
    values = {g: {t: {m: {c: np.array([rows[s][t][g][m] for s in seqs]) for c, seqs in valid[g].items() if seqs}
                      for m in METRICS} for t in spec["cases"]} for g in GROUPS}

    def macro(by_city: dict, picks: dict = None):
        """양성이 있는 시퀀스 -> 도시 -> 도시 평균. picks가 있으면 도시 안 재표집 (B,) 배열을 돌려준다."""
        parts = [v.mean() if picks is None else v[picks[c]].mean(axis=1) for c, v in by_city.items()]
        return np.mean(parts, axis=0)

    # 재현 검사: 기존 보고값
    old = READ(OLD_TEST / "summary.json")
    old_dir = READ(DIRECTIONS / "direction_summary.json")["methods"]
    side = READ(SIDE / "result.json")["groups"]
    sanity = dict(ledger_max_score_error=max(errors))
    gaps = {}
    for m in METRICS:
        for t in ("identity", "fixed"):
            gaps[f"{t}_all_{m}"] = abs(macro(values["all"][t][m]) - old["mean"][t][m])
        for g in ("forward_only", "reverse_only"):
            gaps[f"fixed_{g}_{m}"] = abs(macro(values[g]["fixed"][m]) - old_dir["fixed"]["mean_std"][g][m]["mean"])
        for arm in ARMS:
            for k in STARTS:
                t = tag("symmetric", arm, k)
                gaps[f"{t}_all_{m}"] = abs(macro(values["all"][t][m]) - old["per_case"][t]["macro"][m])
            for g in ("forward_only", "reverse_only"):
                mean = np.mean([macro(values[g][tag("symmetric", arm, k)][m]) for k in STARTS])
                gaps[f"{arm}_{g}_{m}"] = abs(mean - old_dir[arm]["mean_std"][g][m]["mean"])
        for g in GROUPS:
            gaps[f"past_fixed_{g}_{m}"] = abs(100 * macro(values[g]["past_fixed"][m])
                                              - side[g]["percent"]["past_same"][m]) / 100
    sanity["max_macro_gap"] = float(max(gaps.values()))
    assert sanity["max_macro_gap"] < 1e-12, max(gaps.items(), key=lambda kv: kv[1])

    arms = {f"symmetric_learned_{a}": [tag("symmetric", a, k) for k in STARTS] for a in ARMS}
    arms.update({f"past_learned_{a}": [tag("past", a, k) for k in STARTS] for a in ARMS})
    arms.update(past_untrained=["past_fixed"], symmetric_untrained=["fixed"], identity=["identity"])
    contrasts = {}
    for a in ARMS:
        contrasts[f"past_minus_symmetric_learned_{a}"] = [(tag("past", a, k), tag("symmetric", a, k)) for k in STARTS]
        contrasts[f"past_learned_{a}_minus_past_untrained"] = [(tag("past", a, k), "past_fixed") for k in STARTS]

    rng = np.random.default_rng(BOOT_SEED)
    result = dict(arms=arms, contrasts={k: [list(p) for p in v] for k, v in contrasts.items()}, groups={})
    for g in GROUPS:
        sizes = {c: len(seqs) for c, seqs in valid[g].items() if seqs}
        picks = {c: rng.integers(0, n, (BOOTSTRAP, n)) for c, n in sizes.items()}
        block = dict(sequences_per_city=sizes, percent={}, differences_pp={})
        for name, tags in arms.items():
            block["percent"][name] = {}
            for m in METRICS:
                per = [100 * float(macro(values[g][t][m])) for t in tags]
                block["percent"][name][m] = dict(mean=float(np.mean(per)), per_start=per,
                                                 sd=float(np.std(per, ddof=1)) if len(per) > 1 else 0.0)
        for name, pairs in contrasts.items():
            block["differences_pp"][name] = {}
            for m in METRICS:
                diff = {c: np.mean([values[g][a][m][c] - values[g][b][m][c] for a, b in pairs], axis=0)
                        for c in sizes}
                boot = 100 * macro(diff, picks)
                block["differences_pp"][name][m] = dict(
                    mean_pp=100 * float(macro(diff)),
                    per_start_pp=[100 * float(macro(values[g][a][m]) - macro(values[g][b][m])) for a, b in pairs],
                    ci95_pp=[float(v) for v in np.percentile(boot, [2.5, 97.5])])
        result["groups"][g] = block
    result.update(
        sanity_checks=sanity,
        reproduction=READ(OUT / "reproduction.json"),
        training={t: {k: READ(Path(c["checkpoint"]).parent / "training_complete.json")[k]
                      for k in ("init_error", "causal_check", "seconds")}
                  for t, c in spec["cases"].items() if c["checkpoint"]},
        aggregation="시험46 중 해당 방향 양성이 있는 시퀀스 -> 도시 -> 양성이 있는 도시(6) 평균 -> 시작점 3개 평균(SD는 시작점 간, ddof=1).",
        bootstrap=dict(replicates=BOOTSTRAP, seed=BOOT_SEED,
                       scheme="도시 안 시퀀스 수 유지 짝지은 복원추출. 시작점은 고정하고 차이는 시작점 평균."),
        test_used_for_selection=False)
    WRITE(OUT / "result.json", result)
    print(json.dumps({g: {a: round(v["AP"]["mean"], 2) for a, v in b["percent"].items()}
                      for g, b in result["groups"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train_all", "test", "test_worker", "analyze"))
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "test_worker":
        test_worker(a.sequence)
    else:
        globals()[a.action]()
