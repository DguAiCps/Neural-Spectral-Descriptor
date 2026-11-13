#!/usr/bin/env python3
"""정렬 교사 학습의 게이트 세기와 문맥 방향 변화를 분리한다."""
from __future__ import annotations

import time

import numpy as np

import _pdv2_diffattn_alignment_teacher as exp


def parts(key: np.ndarray) -> tuple:
    """정규화 키를 원 특징·게이트 세기·단위 문맥으로 분해한다."""
    raw_norm = np.linalg.norm(key[:, :288], axis=1, keepdims=True)
    assert np.all(raw_norm > 1e-8)
    raw = key[:, :288] / raw_norm
    ctx = key[:, 288:] / raw_norm
    alpha = np.linalg.norm(ctx, axis=1, keepdims=True)
    direction = ctx / np.maximum(alpha, 1e-12)
    return raw, alpha, direction


def assemble(raw: np.ndarray, alpha: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """서로 다른 모델의 세기와 방향을 조립하는 진단용 키를 만든다."""
    key = np.concatenate([raw, alpha * direction], axis=1)
    return key / np.maximum(np.linalg.norm(key, axis=1, keepdims=True), 1e-12)


def main() -> None:
    """학습과 검증의 실제 전체 과거 검색에서 교체 개입을 비교한다."""
    spec = exp.activate()
    assert exp.run.READ(exp.OUT / 'status.json')['status'] == 'validation_complete'
    folder = exp.OUT / 'gate_audit'
    folder.mkdir(exist_ok=True)
    fixed = exp.model()
    trained = exp.model(exp.OUT / 'teacher_lr4/epoch8.pth')
    weights = exp.run.cal.fusion_weights(spec['confidence'])[exp.run.cal.FUSIONS.index('aux_half')]
    records = {}
    for role in ('train', 'val'):
        per = {}
        for sequence in spec[role]:
            tick = time.time()
            cache, pos, minimum, _, bank, _, _, _ = exp.run.inputs(role, sequence)
            k0 = exp.run.source.make_keys(fixed, cache['descriptors'], cache['poses'])
            k1 = exp.run.source.make_keys(trained, cache['descriptors'], cache['poses'])
            raw0, alpha0, direction0 = parts(k0)
            raw1, alpha1, direction1 = parts(k1)
            np.testing.assert_allclose(raw0, raw1, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(assemble(raw0, alpha0, direction0), k0, atol=2e-6, rtol=2e-6)
            variants = dict(frozen=k0, learned=k1,
                            direction_only=assemble(raw0, alpha0, direction1),
                            alpha_only=assemble(raw0, alpha1, direction0),
                            raw=raw0)
            if role == 'train':
                q = exp.run.ARR(exp.OUT / 'pools' / sequence.replace('/', '_') / 'pool.npz')['query']
            else:
                q = np.arange(30, len(pos) - 15, dtype=np.int64)
                q = q[minimum[q] <= 5]
            # 학습용 정답 주입 없이 실제 전체 과거 후보를 검색한다.
            hits = {v: [] for v in variants}
            for t in q:
                aux_ids = exp.run.base.prefix_topk(bank[0], int(t), int(t) - 29, 100)
                for name, key in variants.items():
                    ids = exp.run.base.prefix_topk(key, int(t), int(t) - 29, 100)
                    ranking = ids[np.lexsort((ids, -(key[ids] @ key[t])))]
                    kd = np.linalg.norm(pos[ranking[0]] - pos[t])
                    union = np.unique(np.r_[ids, aux_ids])
                    score = weights[0] * exp.run.base.minmax(key[union] @ key[t])
                    score += weights[1] * exp.run.base.minmax(bank[0][union] @ bank[0][t])
                    pred = union[score.argmax()]
                    hits[name].append((bool(kd <= 5), bool(np.linalg.norm(pos[pred] - pos[t]) <= 5)))
            metric = {v: dict(key_R1=float(np.mean([x[0] for x in rows])),
                              pre_R1=float(np.mean([x[1] for x in rows])))
                      for v, rows in hits.items() if rows}
            gate_stats = {name: dict(mean=float(a.mean()), std=float(a.std()),
                                    quantiles=np.quantile(a, [0, .1, .5, .9, 1]).tolist())
                          for name, a in (('frozen', alpha0), ('learned', alpha1))}
            record = dict(queries=len(q), metrics=metric, gates=gate_stats,
                          direction_cosine=float((direction0 * direction1).sum(1).mean()),
                          seconds=time.time() - tick)
            per[sequence] = record
            exp.run.WRITE(folder / (role + '_' + sequence.replace('/', '_') + '.json'), record)
            print('AUDIT', role, sequence, len(q), metric, flush=True)
        groups = {}
        for sequence, row in per.items():
            if not row['queries']:
                continue
            group = 'kitti' if sequence.startswith('kitti') else sequence.split('/')[0]
            groups.setdefault(group, []).append(row)
        grouped = {g: {v: {m: float(np.mean([r['metrics'][v][m] for r in rows]))
                           for m in ('key_R1', 'pre_R1')} for v in variants}
                   for g, rows in groups.items()}
        macro = {v: {m: float(np.mean([r[v][m] for r in grouped.values()]))
                     for m in ('key_R1', 'pre_R1')} for v in variants}
        records[role] = dict(sequences=per, groups=grouped, macro=macro)
    exp.run.WRITE(exp.OUT / 'gate_audit.json', dict(records=records,
        scope='사후 원인 진단. 교사 강학습8에포크 한 조건. 학습은 초기 고정 질의·검증은 전체 재방문 질의에서 실제 과거 검색. 원고 최종 판정이나 설정 선택 결과가 아니다. raw 대조는 신경망 제거 진단이며 미학습 모델과 다르다.',
        source_sha256=exp.run.HASH(exp.Path(__file__)), test_used=False))


if __name__ == '__main__':
    exp.run.configure()
    main()
