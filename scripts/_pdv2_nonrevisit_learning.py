#!/usr/bin/env python3
"""정답 없는 질의의 최종 점수로 기존 DiffAttnConv를 추가 학습한다."""
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
import _review_response_nsd as current
import _pdv2_spectral_keys as source
from _pdv2_matching_mine import cache_path, spec_path
from _pdv2_training_coverage import prefix_distance
from _pdv2_context_accept_lib import fit_score, predict_score
from _pdv2_input_alignment_lib import make_graph
from _train_attention_neighbors import array_hash
from gnn.trainer import InfoNCELoss
from retrieval.nonrevisit_learning import context_plan, context_score, acceptance_loss

OUT = ROOT / "results/pdv2_nonrevisit_learning_20261104"
READ, WRITE, ARR, HASH = current.READ, current.WRITE, current.ARR, current.HASH
base, cal, power, white = current.base, current.cal, current.power, current.white
ARMS = ("retrieval_only", "positive_only", "all_queries")
LAMBDAS = (0., .125, .25, .5, 1.)
EPOCHS = (2, 4)


def configure() -> None:
    """작은 연산의 CPU 경합과 축약 정밀도 차이를 줄인다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare() -> dict:
    """새 학습·평가 전에 비교와 선택 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = READ(target)
        sources = dict(spec["sources"])
        amendments = OUT / "runtime_amendments.json"
        if amendments.exists():
            for change in READ(amendments)["changes"]:
                assert sources[change["path"]] == change["before"]
                sources[change["path"]] = change["after"]
        for name, digest in sources.items():
            assert HASH(ROOT / name) == digest, name
        return spec
    main = READ(power.OUT / "protocol.json")
    initial = source.root(1, "octave288") / "checkpoints" / f"checkpoint_epoch_{main['models']['1']}.pth"
    paths = [Path(__file__), ROOT / "src/retrieval/nonrevisit_learning.py",
             ROOT / "tests/retrieval/test_nonrevisit_learning.py", Path(source.__file__),
             Path(current.adoption.__file__), Path(cal.__file__),
             ROOT / "src/gnn/model.py", ROOT / "src/gnn/geometry_interaction.py",
             ROOT / "src/gnn/trainer.py", ROOT / "src/retrieval/sequence_evidence.py",
             ROOT / "scripts/_pdv2_context_accept_lib.py", ROOT / "scripts/_pdv2_input_alignment_lib.py",
             ROOT / "scripts/_pdv2_training_coverage.py", ROOT / "scripts/_xs_eval_lib.py",
             ROOT / "scripts/_causal_pr_dump.py", initial,
             source.OUT / "remetric_octave288.npy", white.OUT / "statistics.npz"]
    spec = dict(created_unix=time.time(), seed=1, test_used=False,
                train=READ(source.OUT / "protocol.json")["train"], val=main["val"],
                initial=str(initial.relative_to(ROOT)), confidence=main["confidence"]["1"],
                arms=ARMS, epochs=EPOCHS, steps_per_sequence=8, batch_queries=128,
                retrieval_batch=128, learning_rate=1e-5, weight_decay=1e-5, clip=1.,
                lambdas=LAMBDAS, context_scale_floor=.05,
                objective="같은 초기 모델·표본 순서·추가256업데이트. InfoNCE만 / InfoNCE+양성질의 수락BCE / InfoNCE+모든 적격질의 수락BCE. 양성은 min_past<=5, 음성은 >25, 중간은 제외. 승자 정답 여부가 BCE 정답이며 양성 질의 오위치도 음성 라벨.",
                acceptance="현재 스펙트럼 최종점수를 학습8에서 양의 로지스틱 점수로 적합. 동일 시간 대응 문맥 코사인을 학습8 평균/표준편차(하한.05)로 표준화해 더함. BCE 미분은 기존 GNN으로 전달. 추가 신경망 모듈 없음. 추론 혼합계수는 모든 조건에 같은 0/.125/.25/.5/1을 검증에서 비교.",
                data="학습은 현재 모델과 같은 GT 키프레임·GT 시간 그래프8. 검증은 KISS9. 초기 모델의 실제 과거 승자 고정으로 학습. 대조별 새 키로 최종 후보·승자·스펙트럼·문맥 점수를 모두 재계산. 온라인 q+15, 후보 j<=q-30 유지.",
                controls="양성 전용도 같은 배치를 계산하되 비재방문 BCE만 제외. InfoNCE는 초기 과거 지도에서 최근접 정답과 융합 최고 >25m 오답으로 만든 같은 삼중쌍. 시퀀스마다 같은8업데이트. BN/재계량 고정·dropout 없음. 학습된 키는 검색과 판정 양쪽에 사용.",
                selection="각 조건 epoch2/4와 동일5혼합계수 중 검증 최종AP 최대. 동률 작은계수→이른epoch. 동예산 고정epoch4 결과도 보고. 기존 모델도 동일 혼합계수 비교. 평균AP+.1%p, F1/R100 손실<=.1%p, 계열AP 손실<=.5%p가 기존·retrieval_only·positive_only 모두에 성립하면 3시드 확증 후보. 시험·본문·그림 자동 수정 없음.",
                limitations="현재 선택모델에서 시작한 시드1 예비. 학습 중 승자 목록 미갱신. 학습/검증 입력 방식과 학습 장소 편향은 남음. 반복 사용한 검증의 탐색 결과이며 독립 확증 아님. 기본 점수 적합은 학습 장소 내 출력이므로 분포 전이 미보장.",
                sources={str(p.relative_to(ROOT)): HASH(p) for p in paths})
    WRITE(target, spec)
    return spec


def model(checkpoint: Path = None) -> torch.nn.Module:
    """현재 288+128 구조를 불러오고 재계량은 고정한다."""
    spec = READ(OUT / "protocol.json")
    checkpoint = ROOT / spec["initial"] if checkpoint is None else checkpoint
    net = source.factory(288).cuda().eval()
    state = dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])
    omega = torch.from_numpy(np.load(source.OUT / "remetric_octave288.npy")).float()
    if "gnn.key_scale" in state:
        # 추가 학습은 고정 재계량도 저장한다. 기존 checkpoint에는 이 항목이 없다.
        torch.testing.assert_close(state.pop("gnn.key_scale"), omega, rtol=0, atol=0)
    net.load_state_dict(state, strict=True)
    net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(
        np.load(source.OUT / "remetric_octave288.npy")).float().cuda(), requires_grad=False)
    return net


def inputs(role: str, sequence: str) -> tuple:
    """학습8과 검증9만 허용하며 자세·스펙트럼·거리 라벨을 반환한다."""
    spec = READ(OUT / "protocol.json")
    assert role in ("train", "val") and sequence in spec[role]
    if role == "train":
        cp, sp = cache_path(sequence), spec_path(sequence)
        cache = ARR(cp)
        pos = cache["poses"][:, :3, 3].astype(np.float64)
        minimum = prefix_distance(pos)
    else:
        _, _, paths, minimum = white.prior.BASE.exp.old.base.inputs("val", sequence)
        cp, sp = paths["cache"], paths["spec"]
        cache = ARR(cp)
        pos = cache["poses_gt"][:, :3, 3]
    x = white.coefficients(sp)
    wx = white.transform(x, ARR(white.OUT / "statistics.npz")["full"])
    with np.load(sp) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
    bank = base.variant_inputs(x, 8)
    nb, nc = [base.variant_inputs(power.transform(v, .5), 8) for v in (x, wx)]
    xyz = cache["poses"][:, :3, 3]
    if role == "train":
        xyz = xyz.astype(np.float64)
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    return cache, pos, minimum, travel, bank, nb, nc, (cp, sp)


def retrieve(key: np.ndarray, bank: tuple, pos: np.ndarray, minimum: np.ndarray,
             training: bool = False) -> tuple:
    """전체 과거 후보를 새로 검색하고 학습용 삼중쌍도 추출한다."""
    conf = READ(OUT / "protocol.json")["confidence"]
    weights = cal.fusion_weights(conf)[cal.FUSIONS.index("aux_half")]
    q = np.arange(30, len(key) - 15, dtype=np.int64)
    winner, triplets = [], []
    for t in q:
        ids = np.unique(np.r_[base.prefix_topk(key, t, t-29, 100),
                             base.prefix_topk(bank[0], t, t-29, 100)])
        a = base.align_scores(bank[1], bank[2], int(t), ids)
        c = base.column_align_scores(bank[3], int(t), ids)
        s = (weights[0] * base.minmax(key[ids] @ key[t]) +
             weights[1] * base.minmax(bank[0][ids] @ bank[0][t]) +
             weights[2] * base.minmax(a) + weights[3] * base.minmax(c))
        winner.append(ids[np.argmax(s)])
        if training and minimum[t] <= 5:
            neg = np.linalg.norm(pos[ids] - pos[t], axis=1) > 25
            if neg.any():
                positive = np.argmin(np.linalg.norm(pos[:t-29] - pos[t], axis=1))
                negative = ids[np.argmax(np.where(neg, s, -np.inf))]
                triplets.append((t, positive, negative))
    j = np.asarray(winner, np.int64)
    assert (j <= q-30).all()
    row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
               top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1),
               group=np.zeros(len(q), np.int64))
    return row, np.asarray(triplets, np.int64).reshape(-1, 3)


def neural_scores(key: np.ndarray, path: np.ndarray, q: np.ndarray,
                  j: np.ndarray) -> np.ndarray:
    """동일 미분 연산을 평가에서 배치 실행한다."""
    tensor = torch.from_numpy(key)
    parts = []
    with torch.no_grad():
        for start in range(0, len(q), 256):
            plan = context_plan(path, q[start:start+256], j[start:start+256])
            parts.append(context_score(tensor, {k: torch.from_numpy(v) for k, v in plan.items()}).numpy())
    return np.concatenate(parts).astype(np.float64)


def make_pool(sequence: str) -> None:
    """현재 모델의 학습 자료 검색·정답 없는 질의를 저장한다."""
    prepare()
    name = sequence.replace("/", "_")
    folder = OUT / "pools" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "audit.json").exists():
        return
    tick = time.time()
    cache, pos, minimum, travel, bank, nb, nc, paths = inputs("train", sequence)
    graph = make_graph(cache["descriptors"], cache["poses"], np.zeros(len(pos), np.int64))
    torch.save(graph, folder / "graph.pt")
    net = model()
    with torch.no_grad():
        key = F.normalize(net(graph.cuda()).float(), dim=1).cpu().numpy()
    row, triplets = retrieve(key, bank, pos, minimum, True)
    q, j = row["query_idx"], row["j_star"]
    spectral = current.adoption.scores(nb, nc, travel, q, j)["score"]
    context = neural_scores(key, travel, q, j)
    positive = row["min_past_gt_dist"] <= 5
    eligible = positive | (row["min_past_gt_dist"] > 25)
    correct = positive & (row["top1_geo_dist"] <= 5)
    plan = context_plan(travel, q[eligible], j[eligible])
    np.savez_compressed(folder / "pool.npz", **row, spectral=spectral, context=context,
                        eligible=eligible, positive=positive, correct=correct,
                        triplets=triplets, **{"plan_"+k: v for k, v in plan.items()})
    # 학습 자료에서도 q+15에 계산 가능한 키와 시간 점수임을 확인한다.
    checks = []
    for at in (len(q)//3, 2*len(q)//3):
        end = int(q[at]) + 16
        short = source.make_keys(net, cache["descriptors"], cache["poses"], end)
        np.testing.assert_allclose(short[:q[at]+6], key[:q[at]+6], atol=4e-6, rtol=4e-5)
        value = neural_scores(short, travel[:end], q[at:at+1], j[at:at+1])
        np.testing.assert_allclose(value, context[at:at+1], atol=2e-6, rtol=0)
        checks.append(int(q[at]))
    WRITE(folder / "audit.json", dict(sequence=sequence, nodes=len(pos), queries=len(q),
        positive=int(positive.sum()), nonrevisit=int((row["min_past_gt_dist"]>25).sum()),
        correct=int(correct.sum()), triplets=len(triplets), prefix_queries=checks,
        inputs={str(p.relative_to(ROOT)): HASH(p) for p in paths},
        outputs={p.name: HASH(p) for p in (folder / "graph.pt", folder / "pool.npz")},
        seconds=time.time()-tick))
    print("POOL", sequence, len(q), int(positive.sum()), int(eligible.sum()), round(time.time()-tick, 1), flush=True)


def calibration() -> dict:
    """학습8의 기준 점수 적합과 문맥 척도만 고정한다."""
    spec = prepare()
    target = OUT / "calibration.json"
    if target.exists():
        return READ(target)
    xs, ys, cs, ws, sources = [], [], [], [], {}
    for seq in spec["train"]:
        p = OUT / "pools" / seq.replace("/", "_") / "pool.npz"
        r = ARR(p)
        keep = r["eligible"]
        xs.append(r["spectral"][keep, None]); ys.append(r["correct"][keep].astype(np.float64))
        cs.append(r["context"][keep]); ws.append(np.full(int(keep.sum()), 1/keep.sum()))
        sources[str(p.relative_to(ROOT))] = HASH(p)
    x, y, c, w = map(np.concatenate, (xs, ys, cs, ws))
    w /= w.sum()
    fitted = fit_score(x, y, w)
    mean = float(w @ c)
    scale = max(float(np.sqrt(w @ ((c-mean)**2))), spec["context_scale_floor"])
    record = dict(spectral=fitted, context_mean=mean, context_scale=scale,
                  correct_rate=float(w @ y), sources=sources, test_used=False,
                  protocol_sha256=HASH(OUT / "protocol.json"))
    WRITE(target, record)
    print("CALIBRATION", fitted, mean, scale, flush=True)
    return record


def train(arm: str) -> None:
    """같은 추가 학습 예산에서 비재방문 BCE의 유무만 대조한다."""
    spec, fit = prepare(), calibration()
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / "training_complete.json").exists():
        return
    assert not list(folder.glob("epoch*.pth")), "부분 학습을 덮어쓰지 않는다"
    torch.manual_seed(1)
    net = model()
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                 lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    contrastive = InfoNCELoss(temperature=.1)
    updates, logs, hashes = 0, [], {}
    tick = time.time()
    for epoch in range(1, max(EPOCHS)+1):
        for ordinal, seq in enumerate(spec["train"]):
            pooldir = OUT / "pools" / seq.replace("/", "_")
            graph = torch.load(pooldir / "graph.pt", map_location="cuda", weights_only=False)
            pool = ARR(pooldir / "pool.npz")
            mask = pool["eligible"]
            tensors = {k: torch.from_numpy(pool[k][mask]).cuda() for k in ("positive", "correct")}
            plan = {k: torch.from_numpy(pool["plan_"+k]).cuda() for k in ("query", "candidate", "weight")}
            logit = torch.from_numpy(predict_score(pool["spectral"][mask, None], fit["spectral"])).float().cuda()
            triplets = torch.from_numpy(pool["triplets"]).cuda()
            rng = np.random.default_rng(1000*epoch+ordinal)
            terms, batch_hashes = [], []
            for step in range(spec["steps_per_sequence"]):
                chosen = rng.choice(len(logit), spec["batch_queries"], replace=len(logit)<spec["batch_queries"])
                tr = rng.choice(len(triplets), spec["retrieval_batch"], replace=len(triplets)<spec["retrieval_batch"]) if len(triplets) else np.empty(0, np.int64)
                batch_hashes.append(array_hash(np.r_[chosen, tr]))
                rows = torch.from_numpy(chosen).cuda()
                optimizer.zero_grad(set_to_none=True)
                embedding = F.normalize(net(graph), dim=1)
                ids = triplets[torch.from_numpy(tr).cuda()]
                ranking = contrastive(embedding[ids[:, 0]], embedding[ids[:, 1]], embedding[ids[:, 2]]) if len(tr) else embedding.sum()*0.
                ctx = context_score(embedding, {k: v[rows] for k, v in plan.items()})
                logits = logit[rows] + (ctx-fit["context_mean"])/fit["context_scale"]
                bce = acceptance_loss(logits, tensors["correct"][rows], tensors["positive"][rows], arm=="all_queries")
                loss = ranking + (bce if arm != "retrieval_only" else 0.)
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(net.parameters(), spec["clip"])
                assert torch.isfinite(grad)
                optimizer.step()
                updates += 1
                terms.append([float(loss.detach()), float(ranking.detach()), float(bce.detach()), float(grad)])
            # 학습 시에도 출력 차원이 바뀌거나 고정 BN 통계가 갱신되면 실패한다.
            assert embedding.shape[1] == 416
            record = dict(epoch=epoch, sequence=seq, updates=updates,
                          loss_rank_bce_grad=np.mean(terms, axis=0).tolist(), batch_hashes=batch_hashes)
            logs.append(record)
            WRITE(folder / "progress.json", record)
            print("TRAIN", arm, epoch, seq, updates, record["loss_rank_bce_grad"], flush=True)
            del graph, embedding, loss, ranking, ctx, logits, bce, plan, tensors, logit, triplets
            torch.cuda.empty_cache()
        if epoch in EPOCHS:
            path = folder / f"epoch{epoch}.pth"
            torch.save(dict(model_state_dict=net.state_dict(), epoch=epoch, updates=updates,
                            protocol_sha256=HASH(OUT / "protocol.json")), path)
            hashes[path.name] = HASH(path)
    changed = []
    for name, tensor in net.state_dict().items():
        same = torch.equal(tensor.detach().cpu(), initial[name])
        if "running_" in name or "num_batches_tracked" in name or name.endswith("key_scale"):
            assert same, name
        elif not same:
            changed.append(name)
    assert changed
    WRITE(folder / "training_complete.json", dict(arm=arm, updates=updates, checkpoints=hashes,
        records=logs, changed_parameters=changed, fixed_buffers_verified=True,
        seconds=time.time()-tick, test_used=False))


def validate(arm: str, epoch: int, sequence: str) -> None:
    """학습된 키로 검색부터 최종 판정까지 다시 실행한다."""
    spec, fit = prepare(), calibration()
    folder = OUT / arm / f"epoch{epoch}" / "val"
    folder.mkdir(parents=True, exist_ok=True)
    name = sequence.replace("/", "_")
    target = folder / (name+".json")
    if target.exists():
        return
    tick = time.time()
    checkpoint = None if arm == "initial" else OUT / arm / f"epoch{epoch}.pth"
    net = model(checkpoint)
    cache, pos, minimum, travel, bank, nb, nc, paths = inputs("val", sequence)
    key = source.make_keys(net, cache["descriptors"], cache["poses"])
    row, _ = retrieve(key, bank, pos, minimum)
    q, j = row["query_idx"], row["j_star"]
    refpath = ROOT / "results/paper_fixed_threshold_20261101/nsd_val/s1" / (name+".npz")
    reference = ARR(refpath)
    for k in ("query_idx", "min_past_gt_dist"):
        np.testing.assert_array_equal(row[k], reference[k])
    row["group"] = reference["group"]
    spectral = current.adoption.scores(nb, nc, travel, q, j)["score"]
    context = neural_scores(key, travel, q, j)
    regeneration = None
    if arm == "initial":
        keypath = base.source_folder(1, "val", READ(power.OUT / "protocol.json")) / ("keys_"+name+".npz")
        archived = ARR(keypath)
        np.testing.assert_array_equal(archived["scan_ids"], cache["scan_ids"])
        np.testing.assert_allclose(key, archived["embedding"], atol=4e-6, rtol=4e-5)
        same = j == reference["j_star"]
        np.testing.assert_array_equal(spectral[same], reference["score"][same])
        regeneration = dict(key_max_error=float(abs(key-archived["embedding"]).max()),
            changed_winners=int((~same).sum()), changed_query_indices=q[~same].tolist(),
            source_key_sha256=HASH(keypath),
            note="같은 모델의 FP32 키 재생성에서 생기는 미세 순위 차이. 이번 재생성 기준과 원고 보존 원장을 구분한다.")
    baseline = predict_score(spectral[:, None], fit["spectral"])
    residual = (context-fit["context_mean"])/fit["context_scale"]
    scores = {f"lambda_{lam:g}": baseline+lam*residual for lam in LAMBDAS}
    metrics = {}
    for label, score in scores.items():
        metrics[label] = base.independent_metrics(row, score)
        check = white.prior.BASE.exp.metrics(score, row)
        for m in base.METRICS:
            if metrics[label][m] is not None:
                np.testing.assert_allclose(metrics[label][m], check[m], atol=1e-12, rtol=0)
    assert base.independent_metrics(row, spectral) == metrics["lambda_0"]
    checks = []
    for at in (len(q)//3, 2*len(q)//3):
        t, end = int(q[at]), int(q[at])+16
        small = source.make_keys(net, cache["descriptors"], cache["poses"], end)
        np.testing.assert_allclose(small[:t+6], key[:t+6], atol=4e-6, rtol=4e-5)
        sbank = current.adoption.previous.source.base.exp.truncate_bank(bank, end)
        weights = cal.fusion_weights(spec["confidence"])[[cal.FUSIONS.index("aux_half")]]
        assert cal.candidates(small, sbank, weights, t)[0] == j[at]
        partial = neural_scores(small, travel[:end], q[at:at+1], j[at:at+1])
        np.testing.assert_allclose(partial, context[at:at+1], atol=2e-6, rtol=0)
        b, c = [current.adoption.previous.source.base.exp.truncate_bank(v, end) for v in (nb, nc)]
        np.testing.assert_array_equal(current.adoption.scores(b, c, travel[:end], q[at:at+1], j[at:at+1])["score"], spectral[at:at+1])
        checks.append(t)
    ledger = target.with_suffix(".npz")
    np.savez_compressed(ledger, **row, spectral=spectral, context=context, **scores)
    WRITE(target, dict(status="passed", arm=arm, epoch=epoch, sequence=sequence,
        metrics=metrics, prefix_checks=checks, initial_main_exact=(arm=="initial" and regeneration["changed_winners"]==0),
        regeneration=regeneration,
        ledger_sha256=HASH(ledger), checkpoint_sha256=HASH(ROOT / spec["initial"] if checkpoint is None else checkpoint),
        input_sha256={str(p.relative_to(ROOT)): HASH(p) for p in (*paths, refpath)},
        seconds=time.time()-tick))
    print("VALIDATION", arm, epoch, sequence, round(time.time()-tick, 1), flush=True)


def finish() -> None:
    """최종 지표만으로 조건을 고르고 동일 예산 결과도 함께 저장한다."""
    spec = prepare()
    all_results, selected, fixed = {}, {}, {}
    for arm in ("initial", *ARMS):
        options = []
        for epoch in ((0,) if arm=="initial" else EPOCHS):
            per = {f"lambda_{lam:g}": {} for lam in LAMBDAS}
            for seq in spec["val"]:
                p = OUT / arm / f"epoch{epoch}" / "val" / (seq.replace("/", "_")+".json")
                record = READ(p)
                assert record["status"] == "passed" and HASH(p.with_suffix(".npz")) == record["ledger_sha256"]
                for label in per:
                    per[label][seq] = record["metrics"][label]
            for lam in LAMBDAS:
                result = dict(arm=arm, epoch=epoch, coefficient=lam,
                              **base.macro(per[f"lambda_{lam:g}"]), sequences=per[f"lambda_{lam:g}"])
                all_results[f"{arm}__epoch{epoch}__lambda{lam:g}"] = result
                options.append(result)
        key = lambda r: (-round(r["macro"]["AP"], 12), r["coefficient"], r["epoch"])
        selected[arm] = sorted(options, key=key)[0]
        fixed[arm] = sorted([r for r in options if r["epoch"]==(0 if arm=="initial" else max(EPOCHS))], key=key)[0]
    baseline = all_results["initial__epoch0__lambda0"]
    now = selected["all_queries"]
    checks = {}
    for label, ref in (("main", baseline), ("frozen_context", selected["initial"]),
                       ("retrieval_only", selected["retrieval_only"]), ("positive_only", selected["positive_only"])):
        checks[label] = dict(AP=now["macro"]["AP"]>=ref["macro"]["AP"]+.001,
            F1=now["macro"]["max_F1"]>=ref["macro"]["max_F1"]-.001,
            R100=now["macro"]["R@100P"]>=ref["macro"]["R@100P"]-.001,
            families=all(now["families"][f]["AP"]>=ref["families"][f]["AP"]-.005 for f in now["families"]))
    train_records = {arm: READ(OUT / arm / "training_complete.json") for arm in ARMS}
    assert len({r["updates"] for r in train_records.values()}) == 1
    orders = [[r["batch_hashes"] for r in train_records[arm]["records"]] for arm in ARMS]
    assert orders[0] == orders[1] == orders[2]
    WRITE(OUT / "summary.json", dict(selected=selected, fixed_epoch=fixed, all_results=all_results,
        baseline=baseline, advancement_checks=checks,
        eligible_for_three_seed_confirmation=all(all(v.values()) for v in checks.values()),
        equal_updates_and_samples=True, test_used=False, protocol_sha256=HASH(OUT / "protocol.json")))
    WRITE(OUT / "status.json", dict(status="complete", train_conditions=3, validation_jobs=63, test_used=False))
    print("SUMMARY", {a: (v["epoch"], v["coefficient"], v["macro"]) for a,v in selected.items()}, flush=True)


def run() -> None:
    """자료 준비·동일 학습·검증을 순차 실행하고 실패 상태를 남긴다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    def call(args: list, name: str) -> None:
        with (logs / (name+".log")).open("w") as f:
            subprocess.run([sys.executable, __file__, *args], stdout=f, stderr=subprocess.STDOUT, check=True)
    try:
        for i, seq in enumerate(spec["train"]):
            WRITE(OUT / "status.json", dict(status="running", phase="training_data", done=i, total=8))
            call(["pool", "--sequence", seq], "pool_"+seq.replace("/", "_"))
        calibration()
        for i, arm in enumerate(ARMS):
            WRITE(OUT / "status.json", dict(status="running", phase="training", arm=arm, done=i, total=3))
            call(["train", "--arm", arm], "train_"+arm)
        jobs = [(a, e, s) for a in ("initial", *ARMS) for e in ((0,) if a=="initial" else EPOCHS) for s in spec["val"]]
        def evaluate(job: tuple) -> None:
            arm, epoch, seq = job
            call(["val", "--arm", arm, "--epoch", str(epoch), "--sequence", seq], f"val_{arm}_{epoch}_{seq.replace('/', '_')}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for i, future in enumerate(concurrent.futures.as_completed([pool.submit(evaluate, j) for j in jobs]), 1):
                future.result()
                WRITE(OUT / "status.json", dict(status="running", phase="validation", done=i, total=len(jobs)))
                print("VAL_PROGRESS", i, len(jobs), flush=True)
        finish()
    except Exception as exc:
        WRITE(OUT / "status.json", dict(status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "pool", "calibrate", "train", "val", "finish", "run"))
    p.add_argument("--sequence", default="kitti/01")
    p.add_argument("--arm", choices=("initial", *ARMS), default="all_queries")
    p.add_argument("--epoch", type=int, default=4)
    a = p.parse_args()
    configure()
    if a.action == "prepare": prepare()
    elif a.action == "pool": make_pool(a.sequence)
    elif a.action == "calibrate": calibration()
    elif a.action == "train": train(a.arm)
    elif a.action == "val": validate(a.arm, a.epoch, a.sequence)
    elif a.action == "finish": finish()
    else: run()
