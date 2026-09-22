#!/usr/bin/env python3
"""Paired significance test at the DEPLOYED headline operating point.

Compares per-query success@1 of the headline NSD / NSD-H two-index fusion
(results/per_query_headline/{nsd,nsd_h}) against a Table-1-consistent SC++
reference assembled from:
  * 7 non-NCLT sequences: results/per_query_baselines/sc++/  (already Table-1-exact)
  * 2 NCLT sequences:     results/per_query_headline/scpp_zfix/  (z-corrected fix8)

For each method pair it reports, per sequence and pooled:
  * paired McNemar exact test (b = A-only correct, c = B-only correct)
  * query-weighted gap and its moving-block bootstrap 95% CI + one-sided p
    (blocks of 25/50/100 preserve within-sequence temporal autocorrelation)
  * per-family (KITTI/NCLT/HeLiPR/MulRan) gap CIs

Every method's per-sequence R@1 is checked as an integer hit-count gate against
the frozen Table-1 targets; a mismatch aborts before any statistic is reported.
"""
import argparse
import json, math, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PH = REPO / "results/per_query_headline"
SCPP_BASE = REPO / "results/per_query_baselines/sc++"
SCPP_FILE = {
    "HeLiPR_Town01": "HELIPR_Town01",
    "MulRan_DCC03": "MULRAN_DCC03",
    "MulRan_KAIST03": "MULRAN_KAIST03",
    "MulRan_Riverside03": "MULRAN_Riverside03",
}

# canonical eval sequences and their family
SEQS = ["KITTI_00", "KITTI_05", "KITTI_08", "NCLT_2012-01-08", "NCLT_2013-01-10",
        "HeLiPR_Town01", "MulRan_DCC03", "MulRan_KAIST03", "MulRan_Riverside03"]
FAMILY = {"KITTI_00": "KITTI", "KITTI_05": "KITTI", "KITTI_08": "KITTI",
          "NCLT_2012-01-08": "NCLT", "NCLT_2013-01-10": "NCLT",
          "HeLiPR_Town01": "HeLiPR", "MulRan_DCC03": "MulRan",
          "MulRan_KAIST03": "MulRan", "MulRan_Riverside03": "MulRan"}

# seq -> committed tag used in the branch-fusion JSONs
SEQ2TAG = {"KITTI_00":"K00","KITTI_05":"K05","KITTI_08":"K08",
           "NCLT_2012-01-08":"N12","NCLT_2013-01-10":"N13","HeLiPR_Town01":"TOWN",
           "MulRan_DCC03":"DCC","MulRan_KAIST03":"KAI","MulRan_Riverside03":"RIV"}
NSD_TUPLE = str((0.5, 1.0, 0.0, 0.5))
NSDH_TUPLE = str((1.0, 1.0, 1.0, 2.0))

def _committed_targets():
    """Exact seed1 deployed-tuple R@1 from the committed branch-fusion JSONs
    (the gate my per-query reproduction must reproduce), plus the frozen
    Table-1 SC++ row (single-run analytic; NCLT from the fix8 pipeline)."""
    pb = json.load(open(REPO / "results/height_sota/pbev_branch_fusion.json"))
    hs = json.load(open(REPO / "results/height_sota/hspec_branch_fusion.json"))
    nsd = {s: pb[SEQ2TAG[s]]["variants"]["pbev128_main"][NSD_TUPLE] for s in SEQS}
    nsd_h = {s: hs[SEQ2TAG[s]]["variants"]["hspec12"][NSDH_TUPLE] for s in SEQS}
    scpp = {"KITTI_00":0.962,"KITTI_05":0.942,"KITTI_08":0.843,"NCLT_2012-01-08":0.9149,
            "NCLT_2013-01-10":0.9558,"HeLiPR_Town01":0.521,"MulRan_DCC03":0.788,
            "MulRan_KAIST03":0.998,"MulRan_Riverside03":0.870}
    return {"nsd": nsd, "nsd_h": nsd_h, "scpp": scpp}

T1 = _committed_targets()
ALLOWED_TABLE1_HIT_DELTA = 1


def load_hits(path):
    d = json.load(open(path))
    return {int(r["query_idx"]): int(bool(r["success_at_k1"])) for r in d["records"]}


def load_method(method, seqs, headline_root, scpp_base, scpp_nclt):
    """Return {seq: {qidx: hit}} for nsd / nsd_h / scpp (assembled Table-1)."""
    out = {}
    for seq in seqs:
        if method == "scpp":
            if seq.startswith("NCLT"):
                p = scpp_nclt / f"{seq}.json"
            else:
                p = scpp_base / f"{SCPP_FILE.get(seq, seq)}.json"
        else:
            p = headline_root / method / f"{seq}.json"
        if not p.exists():
            raise FileNotFoundError(f"[{method}] missing {p}")
        out[seq] = load_hits(p)
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided exact binomial p at rate 0.5, evaluated in log space.
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1)
                 - math.lgamma(n - i + 1) - n * math.log(2.0)
                 for i in range(k + 1)]
    m = max(log_terms)
    return min(1.0, 2.0 * math.exp(m) * sum(math.exp(x - m) for x in log_terms))


def block_bootstrap(pairs_by_seq, block, reps, rng):
    """pairs_by_seq: list of (a_hits, b_hits) numpy arrays, ordered by query_idx.
    Moving-block resample within each sequence; return query-weighted gap dist."""
    total = sum(len(a) for a, _ in pairs_by_seq)
    starts_by_seq = [max(1, len(a) - block + 1) for a, _ in pairs_by_seq]
    gaps = np.empty(reps)
    for r in range(reps):
        ha = hb = 0
        for (a, b), nstart in zip(pairs_by_seq, starts_by_seq):
            n = len(a)
            nblk = math.ceil(n / block)
            s = rng.integers(0, nstart, size=nblk)
            idx = np.concatenate([np.arange(x, min(x + block, n)) for x in s])[:n]
            ha += a[idx].sum(); hb += b[idx].sum()
        gaps[r] = (ha - hb) / total
    return gaps


def align(A, B, seqs):
    """Return per-seq ordered (a,b) arrays on the shared query_idx set."""
    pairs, meta = [], []
    for seq in seqs:
        if set(A[seq]) != set(B[seq]):
            raise ValueError(
                f"query-index mismatch for {seq}: {len(A[seq])} vs {len(B[seq])}; "
                "do not intersect or report a partial paired test")
        common = sorted(A[seq])
        a = np.array([A[seq][q] for q in common], dtype=np.int64)
        b = np.array([B[seq][q] for q in common], dtype=np.int64)
        pairs.append((a, b)); meta.append((seq, common))
    return pairs, meta


def validate(method, hits):
    print(f"  [{method}] per-seq hit count vs Table-1:")
    ok = True
    for seq in SEQS:
        observed_hits = int(sum(hits[seq].values()))
        n_queries = len(hits[seq])
        r = observed_hits / max(n_queries, 1)
        t = T1[method][seq]
        target_hits = int(round(t * n_queries))
        hit_delta = observed_hits - target_hits
        flag = "" if abs(hit_delta) <= ALLOWED_TABLE1_HIT_DELTA else "  <-- MISMATCH"
        if flag:
            ok = False
        print(f"    {seq:<20} {r:.4f} (T1 {t:.3f}; hits {observed_hits}/{target_hits}, "
              f"delta {hit_delta:+d}){flag}")
    return ok


def run_pair(nameA, A, nameB, B, reps=10000):
    pairs, meta = align(A, B, SEQS)
    print(f"\n### {nameA} vs {nameB} ###")
    # per-seq McNemar
    perseq = {}
    Bpool = Cpool = hitsA = hitsB = tot = 0
    for (a, b), (seq, common) in zip(pairs, meta):
        bb = int(((a == 1) & (b == 0)).sum())
        cc = int(((a == 0) & (b == 1)).sum())
        p = mcnemar_exact(bb, cc)
        perseq[seq] = {"n": len(common), "rA": float(a.mean()), "rB": float(b.mean()),
                       "b_Aonly": bb, "c_Bonly": cc, "mcnemar_p": p}
        Bpool += bb; Cpool += cc; hitsA += int(a.sum()); hitsB += int(b.sum()); tot += len(a)
        print(f"  {seq:<20} rA={a.mean():.4f} rB={b.mean():.4f} b={bb} c={cc} "
              f"McNemar p={p:.3g}")
    gap = (hitsA - hitsB) / tot
    p_pool = mcnemar_exact(Bpool, Cpool)
    print(f"  POOLED  rA={hitsA/tot:.4f} rB={hitsB/tot:.4f} gap={gap*100:+.2f}pp "
          f"b={Bpool} c={Cpool} McNemar p={p_pool:.3g}")
    # moving-block bootstrap CIs
    boot = {}
    for blk in (25, 50, 100):
        rng = np.random.default_rng(1234 + blk)
        g = block_bootstrap(pairs, blk, reps, rng)
        lo, hi = np.percentile(g, [2.5, 97.5])
        p_lower = (int((g <= 0).sum()) + 1) / (reps + 1)
        p_upper = (int((g >= 0).sum()) + 1) / (reps + 1)
        p_two = min(1.0, 2.0 * min(p_lower, p_upper))
        boot[str(blk)] = {
            "ci95_pp": [lo*100, hi*100],
            "bootstrap_p_two_sided": p_two,
            "is_primary_block": blk == 50,
        }
        print(f"  bootstrap block={blk}: gap95%CI=[{lo*100:+.3f},{hi*100:+.3f}]pp "
              f"two-sided tail p={p_two:.4f}")
    # per-family gap
    fam = {}
    for f in ["KITTI", "NCLT", "HeLiPR", "MulRan"]:
        fseqs = [s for s in SEQS if FAMILY[s] == f]
        fp, _ = align(A, B, fseqs)
        ha = sum(a.sum() for a, _ in fp); hb = sum(b.sum() for _, b in fp)
        n = sum(len(a) for a, _ in fp)
        rng = np.random.default_rng(999)
        g = block_bootstrap(fp, 50, reps, rng)
        lo, hi = np.percentile(g, [2.5, 97.5])
        fam[f] = {"rA": ha/n, "rB": hb/n, "gap_pp": (ha-hb)/n*100,
                  "ci95_pp": [lo*100, hi*100], "n": n}
        print(f"  family {f:<7} rA={ha/n:.4f} rB={hb/n:.4f} gap={(ha-hb)/n*100:+.2f}pp "
              f"CI=[{lo*100:+.2f},{hi*100:+.2f}]")
    return {"pooled": {"rA": hitsA/tot, "rB": hitsB/tot, "gap_pp": gap*100,
                       "b_Aonly": Bpool, "c_Bonly": Cpool, "mcnemar_p": p_pool,
                       "n": tot},
            "per_seq": perseq, "bootstrap": boot, "per_family": fam}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-root", type=Path, default=PH,
                        help="Directory containing nsd/ and nsd_h/ per-query dumps")
    parser.add_argument("--scpp-base", type=Path, default=SCPP_BASE,
                        help="Table-1-consistent SC++ dumps for non-NCLT sequences")
    parser.add_argument("--scpp-nclt", type=Path, default=PH / "scpp_zfix",
                        help="Table-1-consistent corrected-cache SC++ NCLT dumps")
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/paired_headline_stats.json")
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--skip-table1-gate", action="store_true",
                        help="For non-seed-1 sensitivity runs only; still requires exact query sets")
    args = parser.parse_args()

    nsd = load_method("nsd", SEQS, args.headline_root, args.scpp_base, args.scpp_nclt)
    nsd_h = load_method("nsd_h", SEQS, args.headline_root, args.scpp_base, args.scpp_nclt)
    scpp = load_method("scpp", SEQS, args.headline_root, args.scpp_base, args.scpp_nclt)
    if args.skip_table1_gate:
        print("== QUERY-SET VALIDATION ONLY (non-seed-1 sensitivity run) ==")
    else:
        print("== VALIDATION (must match Table-1 before any statistic) ==")
        ok = all([validate("nsd", nsd), validate("nsd_h", nsd_h), validate("scpp", scpp)])
        if not ok:
            print("\nABORT: a per-seq R@1 does not match Table-1. Fix reproduction first.")
            sys.exit(2)
        print("\nAll methods match Table-1 within tolerance. Running paired tests.\n")
    out = {"analysis_protocol": {
               "primary_block": 50,
               "sensitivity_blocks": [25, 100],
               "bootstrap_replicates": args.reps,
               "mcnemar_role": "descriptive; temporal dependence is handled by block bootstrap",
               "table1_hit_delta_limit": ALLOWED_TABLE1_HIT_DELTA,
           },
           "nsd_vs_scpp": run_pair("NSD", nsd, "SC++", scpp, args.reps),
           "nsd_h_vs_scpp": run_pair("NSD-H", nsd_h, "SC++", scpp, args.reps)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(f"\nWROTE {args.output}")


if __name__ == "__main__":
    main()
