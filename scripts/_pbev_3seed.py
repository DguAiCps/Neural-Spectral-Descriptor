#!/usr/bin/env python3
"""3-seed aggregation of the pbev branch-substitution run.
Per seed: select fusion weight on the 6 train tags only (frozen), then report
query-weighted val aggregates. Prints mean +/- std across seeds."""
import json
import statistics
from pathlib import Path

R = Path('/rise/RISE1/workspace/impl/NSD-paper/results/height_sota')
TRAIN = ['K01_TR', 'K02_TR', 'K06_TR', 'K07_TR', 'N0511_TR', 'N0804_TR']
VAL9 = ['K00', 'K05', 'K08', 'N12', 'N13', 'TOWN', 'DCC', 'KAI', 'RIV']
VAL8 = [t for t in VAL9 if t != 'TOWN']
FILES = {1: 'pbev_branch_fusion.json',
         2: 'pbev_branch_fusion_seed2.json',
         3: 'pbev_branch_fusion_seed3.json'}


def agg(rep, variant, tags, w):
    num = den = 0.0
    for t in tags:
        v = rep[t]['variants'][variant]
        num += v['n_queries'] * v[w]
        den += v['n_queries']
    return num / den


for variant in ('pbev128_main', 'pbev192_realloc'):
    rows = {'VAL9': [], 'VAL8': []}
    perseq = {t: [] for t in VAL9}
    wsel = []
    for seed, fn in FILES.items():
        rep = json.loads((R / fn).read_text())
        ws = [k for k in rep[TRAIN[0]]['variants'][variant] if k.startswith('(')]
        w = max(ws, key=lambda x: agg(rep, variant, TRAIN, x))
        wsel.append(w)
        rows['VAL9'].append(agg(rep, variant, VAL9, w))
        rows['VAL8'].append(agg(rep, variant, VAL8, w))
        for t in VAL9:
            perseq[t].append(rep[t]['variants'][variant][w])
    print(f'--- {variant} ---')
    print(f'  weights per seed: {wsel}')
    for scope in ('VAL9', 'VAL8'):
        v = rows[scope]
        print(f'  {scope}: mean={statistics.mean(v):.5f} '
              f'std={statistics.stdev(v):.5f} seeds={[f"{x:.5f}" for x in v]}')
    for t in VAL9:
        v = perseq[t]
        print(f'    {t}: {statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}')
print('THREESEED_DONE')
