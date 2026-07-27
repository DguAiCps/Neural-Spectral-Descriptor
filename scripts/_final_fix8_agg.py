#!/usr/bin/env python3
"""Final corrected-parser aggregation: common train-pooled tuple, 3 seeds,
for hspec6 / hspec12 / pbev128_main. Prints per-seq means and all four paper
aggregates (query-weighted, sequence-mean, sensor-sigma, min sensor mean)."""
import json
import statistics
from pathlib import Path

R = Path('/rise/RISE1/workspace/impl/NSD-paper/results/height_sota')
TRAIN = ['K01_TR', 'K02_TR', 'K06_TR', 'K07_TR', 'N0511_TR', 'N0804_TR']
VAL9 = ['K00', 'K05', 'K08', 'N12', 'N13', 'TOWN', 'DCC', 'KAI', 'RIV']
SENSOR = {'K00': 'kitti', 'K05': 'kitti', 'K08': 'kitti', 'N12': 'nclt',
          'N13': 'nclt', 'TOWN': 'helipr', 'DCC': 'mulran', 'KAI': 'mulran',
          'RIV': 'mulran'}
SETS = {
    'hspec6': ['hspec_branch_fusion.json', 'hspec_branch_fusion_seed2.json',
               'hspec_branch_fusion_seed3.json'],
    'hspec12': ['hspec_branch_fusion.json', 'hspec_branch_fusion_seed2.json',
                'hspec_branch_fusion_seed3.json'],
    'pbev128_main': ['pbev_branch_fusion.json', 'pbev_branch_fusion_seed2.json',
                     'pbev_branch_fusion_seed3.json'],
}


def agg(rep, v, tags, w):
    n = d = 0.0
    for t in tags:
        x = rep[t]['variants'][v]
        n += x['n_queries'] * x[w]
        d += x['n_queries']
    return n / d


for variant, files in SETS.items():
    try:
        reps = [json.loads((R / f).read_text()) for f in files]
    except FileNotFoundError as e:
        print(f'--- {variant}: MISSING {e}')
        continue
    ok = [r for r in reps if all(t in r and variant in r[t].get('variants', {})
                                 for t in TRAIN + VAL9)]
    if len(ok) < len(reps):
        print(f'--- {variant}: only {len(ok)}/3 seeds complete')
    if not ok:
        continue
    ws = [k for k in ok[0][TRAIN[0]]['variants'][variant] if k.startswith('(')]
    w = max(ws, key=lambda x: sum(agg(r, variant, TRAIN, x) for r in ok))
    per = {t: statistics.mean(r[t]['variants'][variant][w] for r in ok)
           for t in VAL9}
    q = {t: ok[0][t]['variants'][variant]['n_queries'] for t in VAL9}
    rq = [agg(r, variant, VAL9, w) for r in ok]
    rs = statistics.mean(per.values())
    sm = {}
    for t in VAL9:
        sm.setdefault(SENSOR[t], []).append(per[t])
    means = [statistics.mean(v) for v in sm.values()]
    mu = statistics.mean(means)
    sigma = (sum((m - mu) ** 2 for m in means) / len(means)) ** 0.5
    print(f'--- {variant} ({len(ok)} seeds) w={w}')
    print(f'  Rq={statistics.mean(rq):.5f} +- '
          f'{statistics.stdev(rq) if len(rq) > 1 else 0:.5f}  '
          f'Rs={rs:.4f}  sigma={sigma:.4f}  Rmin={min(means):.4f}')
    print('  ' + '  '.join(f'{t}={per[t]:.4f}' for t in VAL9))
    print('  q: ' + ' '.join(str(q[t]) for t in VAL9))
print('FINAL_AGG_DONE')
