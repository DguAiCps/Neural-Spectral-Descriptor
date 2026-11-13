#!/usr/bin/env python3
"""Tables, manuscript-integration record and verification for the current-split geography audit.

Inputs (all in results/pdv2_review_closure_20261107/geography/): roster.json, geo_audit.json,
geo_pairs_cross_split.csv. Writes geo_summary_table.tex, geo_pairs_measured_table.tex,
evidence_gaps.json, manuscript_integration.json, verification.json. No input is modified.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/pdv2_review_closure_20261107/geography'
NAME = {'train-val': 'Train--val', 'train-test': 'Train--test', 'val-test': 'Val--test'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(x, nd=2):
    return '--' if x is None else f'{100 * x:.{nd}f}'


def tex_key(k):
    return k.replace('_', r'\_')


def texnum(n):
    return f'{n:,}'.replace(',', '{,}')


def km(x):
    return '--' if x is None else f'{x:,.1f}'.replace(',', '{,}')


def main():
    roster = json.load(open(OUT / 'roster.json'))
    geo = json.load(open(OUT / 'geo_audit.json'))
    rows = list(csv.DictReader(open(OUT / 'geo_pairs_cross_split.csv')))
    S = geo['summary']

    # ---------------- summary table
    L = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
         r'\caption{Geographic separation of the splits over all unordered cross-split sequence pairs. '
         r'\emph{Measured}: both trajectories lie in one GNSS/INS-referenced UTM frame (KITTI and KITTI-360 OXTS, '
         r'MulRan global poses, HeLiPR Town/Roundabout INS; the 54 HeLiPR KAIST/DCC/Riverside pairs are placed by '
         r'registration onto MulRan). The overlap is the largest share of one sequence\textquoteright s ground-truth '
         r'poses, 5\,m route samples or keyframes within 50\,m of the other sequence or of the union of the other '
         r'split. \emph{Bounded}: HeLiPR Bridge, placed only through two public bridge coordinates; the value is the '
         r'worst-case lower bound of the minimum distance. \emph{City level}: different cities; lower bound of the '
         r'minimum distance from city coordinates and route extents. LB: lower bound.}',
         r'\label{tab:split_geo}', r'\begin{tabular}{lrrrrrrrrr}', r'\toprule',
         r' & & \multicolumn{3}{c}{Measured} & \multicolumn{2}{c}{Bounded} & \multicolumn{2}{c}{City level} & \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
         r'Splits & Pairs & $n$ & ${<}50$\,m (\%) & Min (km) & '
         r'$n$ & LB (km) & $n$ & LB (km) & Unverified \\', r'\midrule']
    for k in ('train-val', 'train-test', 'val-test'):
        s = S[k]
        c = s['classes']
        nm = c.get('measured_global', 0) + c.get('measured_registered', 0) + c.get('measured_shared_local', 0)
        vals = [v for v in (s['measured_max_frac_50m'], s['union_max_frac_50m']) if v is not None]
        mx = max(vals) if vals else None
        bm = s['bounded_min_lower_bound_m']
        L.append(f"{NAME[k]} & {s['unordered_pairs']} & {nm} & {pct(mx) if nm else '--'} & "
                 f"{km(s['measured_min_dist_m'] / 1000.0 if s['measured_min_dist_m'] is not None else None)} & "
                 f"{c.get('bounded', 0)} & {km(bm / 1000.0 if bm is not None else None)} & "
                 f"{c.get('geographic_exclusion', 0)} & {km(s['geographic_min_lower_bound_km'])} & "
                 f"{s['unverified_pairs']} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (OUT / 'geo_summary_table.tex').write_text('\n'.join(L) + '\n')

    # ---------------- closest measured cross-split pairs (both directions)
    meas = [r for r in rows if r['comparison'].startswith('measured')]
    und = {}
    for r in meas:
        und.setdefault(tuple(sorted((r['a'], r['b']))), []).append(r)
    top = sorted(und.items(), key=lambda kv: min(float(x['min_dist_m']) for x in kv[1]))[:10]
    T = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
         r'\caption{The ten closest measured cross-split sequence pairs (KITTI/KITTI-360, UTM 32N from OXTS). '
         r'Counts are ground-truth pose samples (10\,Hz) within 50\,m of the other trajectory, in both directions.}',
         r'\label{tab:split_geo_pairs}', r'\begin{tabular}{llrr}', r'\toprule',
         r'Sequence A & Sequence B & Min.\ dist.\ (m) & Within 50\,m (A$\rightarrow$B / B$\rightarrow$A) \\',
         r'\midrule']
    short = {'train': 'train', 'val': 'val', 'test': 'test'}
    for (a, b), rs in top:
        ab = next(x for x in rs if x['a'] == a)
        ba = next(x for x in rs if x['a'] == b)
        dmin = min(float(ab['min_dist_m']), float(ba['min_dist_m']))
        T.append(f"{tex_key(a)} ({short[ab['a_split']]}) & {tex_key(b)} ({short[ab['b_split']]}) & "
                 f"{dmin:,.1f} & {int(ab['count_pose_50m'])}/{int(ab['n_pose']):,} / "
                 f"{int(ba['count_pose_50m'])}/{int(ba['n_pose']):,} \\\\".replace(',', '{,}'))
    T += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (OUT / 'geo_pairs_measured_table.tex').write_text('\n'.join(T) + '\n')

    # ---------------- evidence gaps (what is not an exact measurement and what would make it one)
    fr = geo['frames']
    gaps = []
    bridge = [k for k, v in fr.items() if v['georef'] == 'bounded_public_landmarks']
    if bridge:
        bd = [r for r in rows if r['comparison'] == 'bounded']
        gaps.append(dict(item='HeLiPR Bridge01-04 (train) absolute position', sequences=bridge,
                         affected_pairs=len({tuple(sorted((r['a'], r['b']))) for r in bd}),
                         status='bounded, not exactly measured',
                         min_lower_bound_m=min(float(r['min_dist_lower_bound_m']) for r in bd),
                         missing_file='/mnt/d/NSD_datasets/helipr/Bridge01/Inertial_data/inspva.csv',
                         how_to_obtain='HeLiPR official distribution (request form at '
                                       'https://sites.google.com/view/heliprdataset/download); not requested here '
                                       '(form submission with personal details is outside this task)',
                         effect='an INS fit would replace the bound by an exact distance; the bound already exceeds '
                                'the 50 m rule by more than two orders of magnitude'))
    reg = [k for k, v in fr.items() if v['georef'] == 'registered_to_mulran']
    if reg:
        rr = [r for r in rows if r['comparison'] == 'measured_registered']
        gaps.append(dict(item='HeLiPR KAIST/DCC/Riverside 04-06 (test) absolute position', sequences=reg,
                         affected_pairs=len({tuple(sorted((r['a'], r['b']))) for r in rr}),
                         status='measured after registration onto MulRan global_pose (no own INS on disk)',
                         registration_icp_frac_within_5m={st: geo['evidence']['helipr_georef'][st]['icp_frac_within_5m']
                                                          for st in ('KAIST', 'DCC', 'Riverside')},
                         missing_file='/mnt/d/NSD_datasets/helipr/{KAIST,DCC,Riverside}04/Inertial_data/inspva.csv',
                         how_to_obtain='HeLiPR request form (as above)',
                         effect='cross-split partners are MulRan Sejong / HeLiPR Roundabout (>= 13 km); registration error '
                                'is metres'))
    loc = sorted({v['frame'] for v in fr.values() if v['georef'] == 'shared_local_frame'})
    gaps.append(dict(item='NCLT and MCD have no lat/lon in the GT files', frames=loc,
                     status='geographic exclusion from city metadata (all test-only, no same-city partner in '
                            'another split)',
                     missing_file='none required for the split rule'))
    k09 = fr.get('kitti/09', {}).get('evidence', {}).get('committed_geo_reference_max_dev_m')
    if isinstance(k09, str):
        gaps.append(dict(item='kitti/09 absent from results/kitti_odometry_geo_reference.json',
                         status='recomputed from raw OXTS (kitti_raw_oxts/2011_09_30_drive_0033_sync); '
                                'odometry-pose rigid fit residual checked',
                         missing_file='none'))
    (OUT / 'evidence_gaps.json').write_text(json.dumps(gaps, indent=1, ensure_ascii=False))

    # ---------------- verification
    V = geo['verification']
    rc = roster['checks']
    checks = {
        'roster_all_pass': rc['all_pass'],
        'roster_63_unique_disjoint': rc['counts_ok'] and rc['no_duplicates'] and rc['roles_disjoint'],
        'nclt_9_dates_exact': rc['nclt_ok'], 'mcd_12_names_exact': rc['mcd_ok'],
        'config_matches_except_mcd_comment': rc['config_ok_except_mcd_comment'],
        'cache_hashes_match_ledgers': not rc['cache_hash_mismatches'],
        'test_queries_195726': rc['test_query_total_ok'],
        'cross_split_directed_pairs': len(rows),
        'cross_split_directed_pairs_expected': 2 * (8 * 9 + 8 * 46 + 9 * 46),
        'brute_force_ok': V['brute_force_ok'],
        'reverse_direction_max_min_dist_gap_m': V['reverse_direction_max_min_dist_gap_m'],
        'keyframe_mapping_max_rigid_resid_m': V['keyframe_mapping_max_rigid_resid_m'],
        'same_split_overlap_groups_ok': geo['same_split_groups_ok'],
        'rule_pass_by_role_pair': {k: S[k]['rule_pass'] for k in S},
        'unverified_pairs': sum(S[k]['unverified_pairs'] for k in S),
    }
    checks['pair_count_ok'] = checks['cross_split_directed_pairs'] == checks['cross_split_directed_pairs_expected']
    scope = {
        'verified_exactly': 'roster facts (names, roles, counts, cache files and hashes, ledger rows) and '
                            'measured distances for pairs in a common GNSS/INS-referenced frame',
        'verified_by_bound': 'HeLiPR Bridge (train) against all Korean val/test sequences',
        'verified_by_geography': 'pairs in different cities / continents (lower bound from coordinates and extents)',
        'not_claimed': 'LiDAR field-of-view overlap beyond trajectory distance (80/100/200 m fractions are '
                       'reported only as auxiliary); development exposure of the test sequences is a separate issue'}
    status = 'passed' if (checks['roster_all_pass'] and checks['pair_count_ok'] and checks['brute_force_ok']
                          and checks['same_split_overlap_groups_ok'] and all(checks['rule_pass_by_role_pair'].values())
                          and checks['unverified_pairs'] == 0) else 'see_checks'
    files = ['roster.json', 'roster.csv', 'roster_table.tex', 'geo_audit.json', 'geo_pairs_cross_split.csv',
             'geo_same_split_pairs.csv', 'geo_per_sequence.csv', 'geo_summary_table.tex',
             'geo_pairs_measured_table.tex', 'evidence_gaps.json']
    hashes = {f: sha(OUT / f) for f in files}
    ver = dict(date='2026-11-07', status=status, checks=checks, scope=scope, file_sha256=hashes,
               scripts={p: sha(REPO / p) for p in ('scripts/split_audit/review_v23_20261107_roster.py',
                                                   'scripts/split_audit/review_v23_20261107_geo.py',
                                                   'scripts/split_audit/review_v23_20261107_report.py')})
    (OUT / 'verification.json').write_text(json.dumps(ver, indent=1, ensure_ascii=False))

    # ---------------- manuscript integration
    tv, tt, vt = S['train-val'], S['train-test'], S['val-test']
    allmax = max(x for s in S.values() for x in (s['measured_max_frac_50m'], s['union_max_frac_50m']) if x is not None)
    allmax_where = (max(((s['measured_max_frac_50m'] or 0, s['measured_max_frac_50m_pair']) for s in S.values()))[1]
                    if allmax > 0 else 'none (every measured cross-split pair has 0 samples within 50 m)')
    mseq = sorted({r['a'] for r in meas})
    n_pose_meas = sum(int(geo['frames'][k]['n_pose']) for k in mseq)
    n_kf_meas = sum(int(geo['frames'][k]['n_kf']) for k in mseq)
    cnt50 = sum(int(r['count_pose_50m']) for r in meas)
    aux = {f'max_frac_pose_{t}m': max(float(r[f'frac_pose_{t}m']) for r in meas) for t in (25, 50, 80, 100, 200)}
    near = min((s['measured_min_dist_m'], s['measured_min_dist_pair']) for s in S.values()
               if s['measured_min_dist_m'] is not None)
    bmin = min(s['bounded_min_lower_bound_m'] for s in S.values() if s['bounded_min_lower_bound_m'] is not None)
    gmin = min(s['geographic_min_lower_bound_km'] for s in S.values() if s['geographic_min_lower_bound_km'] is not None)
    nums = dict(n_train=8, n_val=9, n_test=46, n_test_queries=rc['test_query_total'],
                cross_split_unordered_pairs=sum(s['unordered_pairs'] for s in S.values()),
                classes={k: S[k]['classes'] for k in S},
                max_fraction_within_50m=allmax, max_fraction_pair=allmax_where,
                min_measured_distance_m=near[0], min_measured_distance_pair=near[1],
                bridge_min_lower_bound_m=bmin, geographic_min_lower_bound_km=gmin,
                unverified_pairs=checks['unverified_pairs'],
                measured_sequences=len(mseq), measured_sequence_pose_samples=n_pose_meas,
                measured_sequence_keyframes=n_kf_meas, measured_pose_samples_within_50m=cnt50,
                measured_aux_thresholds=aux)
    nreg = sum(S[k]['classes'].get('measured_registered', 0) for k in S)
    sent = [
        ("The 63 sequences form 8 training, 9 validation and 46 test sequences (Table~\\ref{tab:split_roster}); "
         "no sequence is assigned to two splits."),
        (f"For the {sum(S[k]['classes'].get('measured_global', 0) + S[k]['classes'].get('measured_registered', 0) for k in S)} "
         f"cross-split pairs that share a GNSS/INS-referenced UTM frame ({nreg} of them through registration of "
         f"HeLiPR KAIST/DCC/Riverside onto MulRan), none of the {texnum(n_pose_meas)} ground-truth poses "
         f"or {texnum(n_kf_meas)} keyframes of the {len(mseq)} involved sequences lies within 50\\,m of a sequence "
         f"from another split; the closest such trajectories, KITTI~01 (training) and KITTI-360~0000 (test), are "
         f"{near[0] / 1000:.2f}\\,km apart."),
        (f"HeLiPR Bridge has no INS record in the distributed data and is placed through two public bridge "
         f"coordinates; over all placements consistent with them, it remains at least {bmin / 1000:.1f}\\,km from "
         f"every Korean validation and test route."),
        (f"All other cross-split pairs are recorded in different cities, at least {texnum(round(gmin))}\\,km apart."),
        ("Distances are horizontal and computed by exact nearest-neighbour search against trajectories densified "
         "to 0.5\\,m; per-session origin-shifted cache poses are mapped back to the global trajectories before "
         "comparison."),
    ]
    tex = {f: (OUT / f).read_text() for f in ('roster_table.tex', 'geo_summary_table.tex', 'geo_pairs_measured_table.tex')}
    mi = dict(date='2026-11-07', owner='geography worker (claude-opus-5-5)', status=status,
              numbers=nums, sentences_en=sent, latex=tex,
              replacement_proposals=[
                  dict(target='docs/paper/body_v2.3/cvpr2027_body.tex (paragraph "Data and geographic separation")',
                       target_sha256_at_proposal=sha(REPO / 'docs/paper/body_v2.3/cvpr2027_body.tex'),
                       old=('Geographic separation allows at most 1\\% of a route within 50\\,m of a different split; '
                            'overlapping sequences belong to the same split.'),
                       new=('Splits are geographically separated: at most 1\\% of a route may lie within 50\\,m of '
                            'another split, and overlapping sequences share a split. On GNSS/INS-referenced '
                            'trajectories, no pose lies within 50\\,m of another split, and the closest cross-split '
                            f'routes are {near[0] / 1000:.2f}\\,km apart. HeLiPR Bridge, which lacks INS data, remains '
                            f'at least {bmin / 1000:.1f}\\,km from all Korean validation and test routes; all other '
                            'cross-split pairs lie in different cities (Supplementary Table~\\ref{tab:split_geo}).')),
                  dict(target='docs/paper/body_v2.3/cvpr2027_supplement_body.tex (new section after '
                              '"Statistical and geographic breakdowns", or inside it)',
                       target_sha256_at_proposal=sha(REPO / 'docs/paper/body_v2.3/cvpr2027_supplement_body.tex'),
                       new_section_title='Split roster and geographic separation',
                       insert_latex=['roster_table.tex', 'geo_summary_table.tex', 'geo_pairs_measured_table.tex'],
                       paragraph=' '.join(sent[1:])),
                  dict(target='docs/데이터와 분할.md (wiki; root applies)',
                       proposal='split v2 근거 파일을 pd2026 감사 results/split_audit/geo_overlap.json 대신 '
                                'results/pdv2_review_closure_20261107/geography/geo_audit.json으로 바꾸고, '
                                'Bridge는 공개 교량 좌표 기반 하한(INS 없음)임을 적는다')],
              sources={**{f'results/pdv2_review_closure_20261107/geography/{f}': h for f, h in hashes.items()},
                       'results/pdv2_diffattn_contrast_test_20261104/protocol.json':
                           sha(REPO / 'results/pdv2_diffattn_contrast_test_20261104/protocol.json')})
    (OUT / 'manuscript_integration.json').write_text(json.dumps(mi, indent=1, ensure_ascii=False))
    print(json.dumps(dict(status=status, checks=checks, numbers=nums), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
