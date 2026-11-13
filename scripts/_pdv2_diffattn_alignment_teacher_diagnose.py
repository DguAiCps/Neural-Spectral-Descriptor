#!/usr/bin/env python3
"""고정 학습 후보에서 순위 목표가 실제로 개선됐는지 따로 확인한다."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import _pdv2_diffattn_alignment_teacher as run
from retrieval.candidate_learning import masked_minmax


def main() -> None:
    """초기 풀의 정답 회수와 학습 손실을 시퀀스 균등 평균한다."""
    spec = run.activate()
    weights = run.run.cal.fusion_weights(spec['confidence'])[run.run.cal.FUSIONS.index('aux_half')]
    candidates = [('frozen', 0, None)] + [(a, e, run.OUT / a / f'epoch{e}.pth') for a in run.ARMS for e in run.EPOCHS]
    results = {}
    for arm, epoch, checkpoint in candidates:
        if checkpoint is not None and not checkpoint.exists():
            continue
        net = run.model(checkpoint)
        per = {}
        for seq in spec['train']:
            folder = run.OUT / 'pools' / seq.replace('/', '_')
            data = run.run.ARR(folder / 'pool.npz')
            if not len(data['query']):
                continue
            graph = torch.load(run.PARENT / 'pools' / seq.replace('/', '_') / 'graph.pt', map_location='cuda', weights_only=False)
            with torch.no_grad():
                embedding = F.normalize(net(graph), dim=1)
                pool = {k: torch.from_numpy(v).cuda() for k, v in data.items()}
                hits, values = [], []
                for start in range(0, len(data['query']), 128):
                    sl = slice(start, start + 128)
                    key = (embedding[pool['query'][sl], None] * embedding[pool['candidates'][sl].clamp_min(0)]).sum(-1)
                    present = pool['present'][sl]
                    pre = weights[0] * masked_minmax(key, present) + weights[1] * masked_minmax(pool['auxiliary'][sl], present)
                    best_key = key.masked_fill(~present, -torch.inf).argmax(1)
                    best_pre = pre.masked_fill(~present, -torch.inf).argmax(1)
                    correct = pool['correct'][sl]
                    hits.append(torch.stack([correct.gather(1, best_key[:, None]).ravel(),
                                             correct.gather(1, best_pre[:, None]).ravel()], 1).float().cpu().numpy())
                    loss, terms = run.alignment_ranking_loss(key, pool['auxiliary'][sl], pool['teacher'][sl], correct,
                        present, pool['eligible'][sl], weights, 'teacher', spec['temperature'])
                    values.append((len(key), {k: float(v) for k, v in terms.items()}))
                total = len(data['query'])
                stats = {k: sum(n * v[k] for n, v in values) / total for k in ('key', 'joint', 'distill')}
                h = np.concatenate(hits).mean(0)
                per[seq] = dict(key_R1=float(h[0]), pre_R1=float(h[1]), **stats)
                del graph, embedding, pool, key, pre, loss, terms
            torch.cuda.empty_cache()
        results[f'{arm}_{epoch}'] = dict(sequences=per,
            mean={m: float(np.mean([r[m] for r in per.values()])) for m in next(iter(per.values()))})
        print('DIAGNOSE', arm, epoch, results[f'{arm}_{epoch}']['mean'], flush=True)
    run.run.WRITE(run.OUT / 'training_diagnostic.json', dict(results=results,
        scope='학습8 중 양성7·고정 초기 후보·시퀀스 균등 평균. 학습에만 주입된 정답 포함. 새로운 장소의 성능이나 최종 지표가 아니다.',
        protocol_sha256=run.run.HASH(run.OUT / 'protocol.json'), source_sha256=run.run.HASH(run.Path(__file__)), test_used=False))


if __name__ == '__main__':
    run.run.configure()
    main()
