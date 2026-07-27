#!/usr/bin/env python3
"""Train-only weight calibration + frozen-weight val aggregation for the
pbev branch-substitution run, with the dedicated-height branch as reference."""
import json
from pathlib import Path

REPO = Path('/rise/RISE1/workspace/impl/NSD-paper')
R = REPO / 'results/height_sota'

TRAIN = ['K01_TR', 'K02_TR', 'K06_TR', 'K07_TR', 'N0511_TR', 'N0804_TR']
VAL9 = ['K00', 'K05', 'K08', 'N12', 'N13', 'TOWN', 'DCC', 'KAI', 'RIV']
VAL8 = [t for t in VAL9 if t != 'TOWN']


def weights_in(vals):
    return [k for k in vals if k.startswith('(')]


def agg(rep, variant, tags, w):
    num = den = 0.0
    for t in tags:
        v = rep[t]['variants'][variant]
        n = v['n_queries']
        num += n * v[w]
        den += n
    return num / den


def calibrate(rep, variant):
    some = rep[TRAIN[0]]['variants'][variant]
    ws = weights_in(some)
    return max(ws, key=lambda w: agg(rep, variant, TRAIN, w))


def report(rep, variant):
    w = calibrate(rep, variant)
    print(f'--- {variant} ---')
    print(f'  train-selected w={w}  (train agg {agg(rep, variant, TRAIN, w):.5f})')
    for scope, tags in (('VAL9', VAL9), ('VAL8', VAL8)):
        print(f'  {scope} Rbar_q = {agg(rep, variant, tags, w):.5f}')
    for t in VAL9:
        v = rep[t]['variants'][variant]
        print(f'    {t}: frozen={v[w]:.4f} nsd={v["nsd"]:.4f} '
              f'h={v["height"]:.4f} oracle={v["oracle_union"]:.4f} q={v["n_queries"]}')
    # strict sketch-only (w3=0) alternative
    ws0 = [x for x in weights_in(rep[TRAIN[0]]['variants'][variant])
           if x.endswith(', 0.0)')]
    w0 = max(ws0, key=lambda x: agg(rep, variant, TRAIN, x))
    print(f'  strict(no-column) w={w0}: VAL9={agg(rep, variant, VAL9, w0):.5f} '
          f'VAL8={agg(rep, variant, VAL8, w0):.5f}')


pb = json.loads((R / 'pbev_branch_fusion.json').read_text())
for v in ('pbev128_main', 'pbev192_realloc'):
    report(pb, v)

ref_path = R / 'height_candidate_fusion.json'
if ref_path.exists():
    hc = json.loads(ref_path.read_text())
    have = [t for t in TRAIN + VAL9 if t in hc]
    for variant in ('raw_height240', 'legacy240'):
        if all(variant in hc[t].get('variants', {}) for t in have):
            if all(t in hc for t in TRAIN + VAL9):
                report(hc, variant)
            else:
                missing = [t for t in TRAIN + VAL9 if t not in hc]
                print(f'--- {variant}: missing tags {missing}, skipping full report')
print('CALIB_DONE')
