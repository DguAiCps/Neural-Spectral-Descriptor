#!/usr/bin/env python3
"""Exact roster of the current split v2 (train 8 / val 9 / test 46) and its cross-checks.

Authority: results/pdv2_diffattn_contrast_test_20261104/protocol.json (train/val/test).
Cross-checked against
  * configs/training_place_disjoint_v2.yaml (MCD is listed only in its comment),
  * the training protocol results/pdv2_diffattn_contrast_seeds_20261104/s{1,2,3}/protocol.json,
  * train pools results/pdv2_diffattn_gain_readout_20261104/pools/train/*/audit.json,
  * val ledgers results/pdv2_diffattn_contrast_retrieval_canonical_20261104/val/*/summary.json,
  * test ledgers results/pdv2_diffattn_contrast_test_20261104/test/*/summary.json,
  * the cache files on disk (sha256 recomputed and compared with each ledger record).
READ-ONLY w.r.t. the repo inputs and /mnt/d. Output: results/pdv2_review_closure_20261107/geography/
roster.{json,csv}, roster_table.tex.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/pdv2_review_closure_20261107/geography'
DS = Path('/mnt/d/NSD_datasets')
PROTOCOL = REPO / 'results/pdv2_diffattn_contrast_test_20261104/protocol.json'
CONFIG = REPO / 'configs/training_place_disjoint_v2.yaml'
TRAIN_PROTOCOLS = [REPO / f'results/pdv2_diffattn_contrast_seeds_20261104/s{i}/protocol.json' for i in (1, 2, 3)]
TRAIN_POOLS = REPO / 'results/pdv2_diffattn_gain_readout_20261104/pools/train'
VAL_LEDGERS = REPO / 'results/pdv2_diffattn_contrast_retrieval_canonical_20261104/val'
TEST_LEDGERS = REPO / 'results/pdv2_diffattn_contrast_test_20261104/test'

# KITTI odometry devkit readme: sequence -> (raw drive, first frame, last frame)
KITTI_RAW = {'00': ('2011_10_03_drive_0027', 0, 4540), '01': ('2011_10_03_drive_0042', 0, 1100),
             '02': ('2011_10_03_drive_0034', 0, 4660), '05': ('2011_09_30_drive_0018', 0, 2760),
             '06': ('2011_09_30_drive_0020', 0, 1100), '07': ('2011_09_30_drive_0027', 0, 1100),
             '08': ('2011_09_30_drive_0028', 1100, 5170), '09': ('2011_09_30_drive_0033', 0, 1590)}
MCD_EXPECTED = ['kth_day_06', 'kth_day_09', 'kth_day_10', 'kth_night_01', 'kth_night_04', 'kth_night_05',
                'tuhh_day_02', 'tuhh_day_03', 'tuhh_day_04', 'tuhh_night_07', 'tuhh_night_08', 'tuhh_night_09']
ROLE_NAME = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}
NCLT_EXPECTED = ['2012-01-08', '2012-02-04', '2012-03-17', '2012-05-11', '2012-08-04', '2012-11-04',
                 '2012-11-16', '2013-01-10', '2013-02-23']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def site_of(ds, seq):
    """(dataset label, site, city, country) for one sequence."""
    if ds == 'kitti':
        return 'KITTI', 'odometry', 'Karlsruhe', 'Germany'
    if ds == 'kitti360':
        return 'KITTI-360', 'suburban drives', 'Karlsruhe', 'Germany'
    if ds == 'nclt':
        return 'NCLT', 'UMich North Campus', 'Ann Arbor, MI', 'USA'
    if ds == 'mcd':
        return ('MCD', 'KTH campus', 'Stockholm', 'Sweden') if seq.startswith('kth') else \
               ('MCD', 'TUHH campus', 'Hamburg', 'Germany')
    site = ''.join(c for c in seq if not c.isdigit())
    if ds == 'mulran':
        city = 'Sejong' if site == 'Sejong' else 'Daejeon'
        return 'MulRan', site, city, 'South Korea'
    city = {'Bridge': 'Seoul', 'Town': 'Seoul', 'Roundabout': 'Ansan'}.get(site, 'Daejeon')
    return 'HeLiPR', site, city, 'South Korea'


def raw_source(ds, seq):
    """Original dataset identifier and the trajectory file used for geography."""
    if ds == 'kitti':
        d, a, b = KITTI_RAW[seq]
        return (f'odometry {seq} = raw {d}_sync frames {a}-{b}',
                DS / f'kitti/dataset/poses/{seq}.txt')
    if ds == 'kitti360':
        return f'2013_05_28_drive_{seq}_sync', DS / f'kitti360/2013_05_28_drive_{seq}_sync/poses.txt'
    if ds == 'nclt':
        return f'NCLT session {seq}', DS / f'nclt/{seq}/groundtruth_{seq}.csv'
    if ds == 'mcd':
        return f'MCD {seq} ({seq}_os1_64.bag)', DS / f'mcd_extracted/{seq}/scan_poses.csv'
    if ds == 'mulran':
        return f'MulRan {seq}', DS / f'mulran/{seq}/global_pose.csv'
    return f'HeLiPR {seq} (Velodyne VLP-16)', DS / f'helipr/{seq}/LiDAR_GT/Velodyne_gt.txt'


def config_roles():
    cfg = yaml.safe_load(open(CONFIG))
    roles = {}
    for role in ('train', 'val', 'test'):
        for e in cfg['data']['datasets'][role]:
            for s in e['sequences']:
                roles.setdefault(f"{e['type']}/{s}", []).append(role)
    return roles


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prot = json.load(open(PROTOCOL))
    split = {r: list(prot[r]) for r in ('train', 'val', 'test')}
    keys = [k for r in ('train', 'val', 'test') for k in split[r]]
    checks = {}
    checks['counts'] = {r: len(split[r]) for r in split}
    checks['counts_ok'] = checks['counts'] == {'train': 8, 'val': 9, 'test': 46}
    checks['no_duplicates'] = len(keys) == len(set(keys)) == 63
    inter = {f'{a}&{b}': sorted(set(split[a]) & set(split[b]))
             for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    checks['role_intersections'] = inter
    checks['roles_disjoint'] = not any(inter.values())
    nclt = sorted(k.split('/')[1] for k in keys if k.startswith('nclt/'))
    mcd = sorted(k.split('/')[1] for k in keys if k.startswith('mcd/'))
    checks['nclt_dates'] = nclt
    checks['nclt_ok'] = nclt == NCLT_EXPECTED
    checks['mcd_names'] = mcd
    checks['mcd_ok'] = mcd == MCD_EXPECTED

    # config yaml (MCD only in a comment -> expected difference, reported explicitly)
    croles = config_roles()
    cfg_diff = []
    for r in split:
        for k in split[r]:
            got = croles.get(k)
            if got != [r]:
                cfg_diff.append({'sequence': k, 'protocol': r, 'config': got})
    for k, v in croles.items():
        if k not in keys:
            cfg_diff.append({'sequence': k, 'protocol': None, 'config': v})
    checks['config_differences'] = cfg_diff
    checks['config_ok_except_mcd_comment'] = all(d['sequence'].startswith('mcd/') and d['config'] is None
                                                  and d['protocol'] == 'test' for d in cfg_diff)
    # training protocols
    tp = {}
    for p in TRAIN_PROTOCOLS:
        if p.exists():
            d = json.load(open(p))
            tp[str(p.relative_to(REPO))] = (d.get('train') == split['train'], d.get('val') == split['val'])
    checks['training_protocols_match'] = tp
    checks['training_protocols_ok'] = bool(tp) and all(a and b for a, b in tp.values())

    rows = []
    hash_mismatch = []
    for role in ('train', 'val', 'test'):
        for k in split[role]:
            ds, seq = k.split('/')
            label, site, city, country = site_of(ds, seq)
            rid, src = raw_source(ds, seq)
            flat = k.replace('/', '_')
            rec = dict(split=role, key=k, dataset=label, sequence=seq, site=site, city=city, country=country,
                       raw_identifier=rid, trajectory_source=str(src), trajectory_source_exists=src.exists())
            rec['trajectory_source_sha256'] = sha256(src) if src.exists() else None
            if role == 'train':
                aud = json.load(open(TRAIN_POOLS / flat / 'audit.json'))
                assert aud['sequence'] == k and aud['role'] == 'train', (k, aud['sequence'], aud['role'])
                ledger, ledger_rows = TRAIN_POOLS / flat / 'audit.json', aud['rows']
                inputs = aud['inputs']
            else:
                led = (VAL_LEDGERS if role == 'val' else TEST_LEDGERS) / flat / 'summary.json'
                s = json.load(open(led))
                aud = s['audit'] if role == 'val' else s['input_audit']
                assert aud['role'] == role, (k, aud['role'])
                ledger, ledger_rows = led, aud['nodes']
                inputs = aud['inputs']
            cache = [p for p in inputs if Path(p).name.startswith('cache_')]
            assert len(cache) == 1, (k, inputs)
            cp = REPO / cache[0]
            now = sha256(cp)
            if now != inputs[cache[0]]:
                hash_mismatch.append((k, cache[0]))
            z = np.load(cp)
            n_cache = int(len(z['timestamps']))
            rec.update(cache=cache[0], cache_sha256=now, cache_sha256_matches_ledger=now == inputs[cache[0]],
                       cache_keyframes=n_cache, ledger=str(ledger.relative_to(REPO)), ledger_rows=int(ledger_rows),
                       ledger_rows_field='pool rows (train)' if role == 'train' else 'nodes',
                       other_inputs={p: h for p, h in inputs.items() if p != cache[0]})
            if role != 'train' and n_cache != int(ledger_rows):
                rec['node_count_mismatch'] = True
            pool = ledger.parent / 'candidate_pool.json'
            rec['ledger_queries'] = int(json.load(open(pool))['queries']) if pool.exists() else None
            rows.append(rec)
    checks['cache_hash_mismatches'] = hash_mismatch
    checks['eval_node_count_mismatches'] = [r['key'] for r in rows if r.get('node_count_mismatch')]
    checks['trajectory_sources_missing'] = [r['key'] for r in rows if not r['trajectory_source_exists']]
    # test ledger directory must contain exactly the 46 test sequences
    led46 = sorted(p.name for p in TEST_LEDGERS.iterdir() if p.is_dir())
    checks['test_ledger_dirs'] = len(led46)
    checks['test_ledger_dirs_ok'] = led46 == sorted(k.replace('/', '_') for k in split['test'])
    led9 = sorted(p.name for p in VAL_LEDGERS.iterdir() if p.is_dir())
    checks['val_ledger_dirs_ok'] = led9 == sorted(k.replace('/', '_') for k in split['val'])
    pools = sorted(p.name for p in TRAIN_POOLS.iterdir() if p.is_dir())
    checks['train_pool_dirs_ok'] = pools == sorted(k.replace('/', '_') for k in split['train'])
    checks['test_node_total'] = sum(r['ledger_rows'] for r in rows if r['split'] == 'test')
    checks['test_query_total'] = sum(r['ledger_queries'] for r in rows if r['split'] == 'test')
    checks['test_query_total_ok'] = checks['test_query_total'] == 195726
    checks['test_queries_equal_nodes_minus_45'] = all(r['ledger_queries'] == r['ledger_rows'] - 45
                                                      for r in rows if r['split'] == 'test')
    ok_keys = ['counts_ok', 'no_duplicates', 'roles_disjoint', 'nclt_ok', 'mcd_ok', 'config_ok_except_mcd_comment',
               'training_protocols_ok', 'test_ledger_dirs_ok', 'val_ledger_dirs_ok', 'train_pool_dirs_ok',
               'test_query_total_ok']
    checks['all_pass'] = all(checks[k] for k in ok_keys) and not hash_mismatch and \
        not checks['eval_node_count_mismatches'] and not checks['trajectory_sources_missing']

    meta = dict(date='2026-11-07', script='scripts/split_audit/review_v23_20261107_roster.py',
                authority=str(PROTOCOL.relative_to(REPO)), authority_sha256=sha256(PROTOCOL),
                config=str(CONFIG.relative_to(REPO)), config_sha256=sha256(CONFIG),
                note_queries='ledger_rows = keyframe nodes; ledger_queries = candidate_pool.json queries '
                             '(test: nodes - 45 per sequence; 195,726 in total)')
    (OUT / 'roster.json').write_text(json.dumps(dict(_meta=meta, checks=checks, roster=rows), indent=1,
                                                ensure_ascii=False))
    cols = ['split', 'key', 'dataset', 'sequence', 'site', 'city', 'country', 'raw_identifier',
            'trajectory_source', 'trajectory_source_sha256', 'cache', 'cache_sha256', 'cache_keyframes',
            'ledger', 'ledger_rows', 'ledger_rows_field', 'ledger_queries']
    with open(OUT / 'roster.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_tex(rows)
    print(json.dumps({k: v for k, v in checks.items() if k not in ('nclt_dates', 'mcd_names')}, indent=1))


def qsum(rs):
    q = [r['ledger_queries'] for r in rs]
    return f'{sum(q):,}' if all(v is not None for v in q) else '--'


def write_tex(rows):
    """Paper table: one line per dataset/site group and split, exact sequence names."""
    groups = {}
    for r in rows:
        g = (r['split'], r['dataset'], r['site'] if r['dataset'] in ('HeLiPR', 'MulRan', 'MCD') else '')
        groups.setdefault(g, []).append(r)
    order = {'train': 0, 'val': 1, 'test': 2}
    lines = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\caption{Exact sequence roster of the place-disjoint split (8 training, 9 validation and '
             r'46 test sequences). City is the recording city. Keyframes are the rows of the cache read by '
             r'training or by the frozen ledgers; queries are the test queries with a retrieval window '
             r'(195{,}726 in total).}',
             r'\label{tab:split_roster}', r'\begin{tabular}{lllp{0.33\linewidth}rr}', r'\toprule',
             r'Split & Dataset (site) & City & Sequences & Keyframes & Queries \\', r'\midrule']
    last = None
    for g in sorted(groups, key=lambda g: (order[g[0]], [r['key'] for r in rows].index(groups[g][0]['key']))):
        rs = groups[g]
        role, ds, site = g
        if last is not None and role != last:
            lines.append(r'\midrule')
        last = role
        names = ', '.join(r['sequence'].replace('_', r'\_') for r in rs)
        dsn = f'{ds} ({site})' if site else ds
        lines.append(f"{ROLE_NAME[role]} & {dsn} & {rs[0]['city']} & "
                     f"\\raggedright {names} & {sum(r['cache_keyframes'] for r in rs):,} & "
                     f"{qsum(rs)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    tex = '\n'.join(lines)
    # thousands separator only in the keyframe column
    out = []
    for ln in tex.split('\n'):
        if ln.endswith(r'\\') and '&' in ln and not ln.startswith('Split'):
            parts = ln.split('&')
            ln = '&'.join(parts[:-2] + [' ' + parts[-2].strip().replace(',', '{,}') + ' ',
                                        ' ' + parts[-1].strip().replace(',', '{,}')])
        out.append(ln)
    (OUT / 'roster_table.tex').write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
