#!/usr/bin/env python3
"""Per-sensor train-only weight calibration for the pbev substitution.
Sensors with train tags (KITTI, NCLT) get their own weight; sensors without
(MulRan, HeLiPR) keep the globally train-selected weight. 3-seed."""
import json
import statistics
from pathlib import Path

R = Path('/rise/RISE1/workspace/impl/NSD-paper/results/height_sota')
FILES = {1: 'pbev_branch_fusion.json',
         2: 'pbev_branch_fusion_seed2.json',
         3: 'pbev_branch_fusion_seed3.json'}
TRAIN_ALL = ['K01_TR', 'K02_TR', 'K06_TR', 'K07_TR', 'N0511_TR', 'N0804_TR']
TRAIN_BY = {'kitti': ['K01_TR', 'K02_TR', 'K06_TR', 'K07_TR'],
            'nclt': ['N0511_TR', 'N0804_TR']}
VAL9 = ['K00', 'K05', 'K08', 'N12', 'N13', 'TOWN', 'DCC', 'KAI', 'RIV']
SENSOR = {'K00': 'kitti', 'K05': 'kitti', 'K08': 'kitti',
          'N12': 'nclt', 'N13': 'nclt', 'TOWN': 'helipr',
          'DCC': 'mulran', 'KAI': 'mulran', 'RIV': 'mulran'}


def agg(rep, variant, tags, w):
    num = den = 0.0
    for t in tags:
        v = rep[t]['variants'][variant]
        num += v['n_queries'] * v[w]
        den += v['n_queries']
    return num / den


for variant in ('pbev128_main', 'pbev192_realloc'):
    vals9 = []
    perseq = {t: [] for t in VAL9}
    for seed, fn in FILES.items():
        rep = json.loads((R / fn).read_text())
        ws = [k for k in rep[TRAIN_ALL[0]]['variants'][variant] if k.startswith('(')]
        wglob = max(ws, key=lambda x: agg(rep, variant, TRAIN_ALL, x))
        wsen = {s: max(ws, key=lambda x: agg(rep, variant, tags, x))
                for s, tags in TRAIN_BY.items()}
        if seed == 1:
            print(f'--- {variant} --- global={wglob} kitti={wsen["kitti"]} nclt={wsen["nclt"]}')
        num = den = 0.0
        for t in VAL9:
            w = wsen.get(SENSOR[t], wglob)
            v = rep[t]['variants'][variant]
            num += v['n_queries'] * v[w]
            den += v['n_queries']
            perseq[t].append(v[w])
        vals9.append(num / den)
    print(f'  per-sensor VAL9: mean={statistics.mean(vals9):.5f} '
          f'std={statistics.stdev(vals9):.5f}')
    for t in VAL9:
        print(f'    {t}: {statistics.mean(perseq[t]):.4f}')
print('PERSENSOR_DONE')
