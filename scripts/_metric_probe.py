#!/usr/bin/env python3
"""Step-1 probe: closed-form re-metrization of the frozen 288D magnitude key.

Fits diagonal-Fisher / diagonal-WCCN(control) / full-WCCN metrics on TRAIN
revisit pairs, then evaluates d-only Recall@1 / R_min / same-place spread on VAL.
Zero training, host numpy only. Tests whether re-metrizing d beats the frozen
baseline BEFORE committing to a learned head + compaction loss (step 2).
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/preprocessed_cross_sensor_operating"
VAL = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DTH, SKIP, EPS = 5.0, 30, 0.1


def load(sensor, seq):
    d = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
    return d["descriptors"].astype(np.float64), d["poses"][:, :3, 3].astype(np.float64)


def norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# ---------- fit metrics on TRAIN revisit pairs ----------
def harvest_train():
    deltas, alld = [], []
    files = sorted(glob.glob(str(CACHE / "*_operating_*_layout60_stride1.npz")))
    valset = {(s, q) for s, qs in VAL.items() for q in qs}
    for f in files:
        base = Path(f).name.replace("_operating_", "|").replace("_layout60_stride1.npz", "")
        sensor, seq = base.split("|")
        if (sensor, seq) in valset:
            continue
        d = np.load(f)["descriptors"].astype(np.float64)
        pos = np.load(f)["poses"][:, :3, 3].astype(np.float64)
        alld.append(d)
        n = len(d)
        for q in range(SKIP, n):
            prior = np.arange(0, q - SKIP)
            if prior.size == 0:
                continue
            geo = np.linalg.norm(pos[prior] - pos[q], axis=1)
            positives = prior[geo < DTH]
            if positives.size == 0:
                continue
            m = positives[np.argmin(np.linalg.norm(pos[positives] - pos[q], axis=1))]
            deltas.append(d[q] - d[m])
    return np.array(deltas), np.concatenate(alld, 0)


def fit_metrics(deltas, alld):
    Wdiag = 0.5 * np.mean(deltas ** 2, axis=0)          # within-place per-coord var
    Tdiag = np.var(alld, axis=0)                         # total per-coord var
    Bdiag = np.maximum(Tdiag - Wdiag, 1e-9)             # between-place per-coord var
    eps = 1e-6 * Wdiag.mean()
    r_fisher = np.sqrt(Bdiag / (Wdiag + eps))
    r_fisher /= r_fisher.mean()
    r_wccn = 1.0 / np.sqrt(Wdiag + eps)                 # pure within-whitening (control)
    r_wccn /= r_wccn.mean()
    # full within-cov WCCN with shrinkage
    Sw = 0.5 * (deltas.T @ deltas) / len(deltas)
    sh = 0.1
    Sw = (1 - sh) * Sw + sh * np.eye(Sw.shape[0]) * np.trace(Sw) / Sw.shape[0]
    ev, evec = np.linalg.eigh(Sw)
    ev = np.maximum(ev, 1e-8)
    L = evec @ np.diag(ev ** -0.5) @ evec.T             # Sw^{-1/2}
    mu = alld.mean(0)
    return {"r_fisher": r_fisher, "r_wccn": r_wccn, "L_wccn": L, "mu": mu}


# ---------- eval d-only on VAL ----------
def eval_seq(d, pos):
    r = norm(d); n = len(d)
    correct = same_d = 0; nq = 0; spreads = []; sp_over = 0
    for q in range(n):
        mask = np.abs(np.arange(n) - q) >= SKIP
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        geo = np.linalg.norm(pos[idx] - pos[q], axis=1)
        positives = idx[geo < DTH]
        if positives.size == 0:
            continue
        nq += 1
        dd = np.linalg.norm(r[idx] - r[q], axis=1)
        nn = idx[dd.argmin()]
        if np.linalg.norm(pos[nn] - pos[q]) < DTH:
            correct += 1
        bp = np.linalg.norm(r[positives] - r[q], axis=1).min()
        spreads.append(bp)
        if bp > EPS:
            sp_over += 1
    return correct / max(nq, 1), nq, float(np.median(spreads)), 100.0 * sp_over / max(nq, 1)


def apply_metric(d, name, M):
    if name == "baseline":
        return d
    if name == "diag_fisher":
        return d * M["r_fisher"]
    if name == "diag_wccn":
        return d * M["r_wccn"]
    if name == "full_wccn":
        return (d - M["mu"]) @ M["L_wccn"].T
    raise ValueError(name)


def summarize(per_seq):
    QC = {"kitti/00": 632, "kitti/05": 377, "kitti/08": 235, "nclt/2012-01-08": 1834,
          "nclt/2013-01-10": 181, "helipr/Town01": 1586, "mulran/DCC03": 2344,
          "mulran/KAIST03": 2909, "mulran/Riverside03": 2796}
    ks = list(per_seq.keys())
    rq = sum(per_seq[k]["r1"] * QC[k] for k in ks) / sum(QC[k] for k in ks)
    rs = float(np.mean([per_seq[k]["r1"] for k in ks]))
    sens = {}
    for k in ks:
        sens.setdefault(k.split("/")[0], []).append(per_seq[k]["r1"])
    macro = {s: float(np.mean(v)) for s, v in sens.items()}
    mbar = np.mean(list(macro.values()))
    sigma = float(np.sqrt(np.sum([(macro[s] - mbar) ** 2 for s in macro]) / (len(macro) - 1)))
    return {"R^q": round(rq, 4), "R^s": round(rs, 4), "sigma_cross": round(sigma, 4),
            "R_min": round(min(macro.values()), 4), "macro": {s: round(v, 4) for s, v in macro.items()}}


def main():
    print("harvesting TRAIN revisit pairs...", flush=True)
    deltas, alld = harvest_train()
    print(f"  {len(deltas)} within-place pairs, {len(alld)} train keyframes", flush=True)
    M = fit_metrics(deltas, alld)

    report = {}
    for name in ["baseline", "diag_fisher", "diag_wccn", "full_wccn"]:
        per_seq = {}
        for sensor, seqs in VAL.items():
            for seq in seqs:
                d, pos = load(sensor, seq)
                dm = apply_metric(d, name, M)
                r1, nq, spmed, sprate = eval_seq(dm, pos)
                per_seq[f"{sensor}/{seq}"] = {"r1": r1, "sameplace_med": round(spmed, 4), "SpreadRate": round(sprate, 1)}
        summ = summarize(per_seq)
        report[name] = {"per_seq": per_seq, "summary": summ}
        print(f"\n=== {name} ===")
        print(f"  R^q={summ['R^q']} R^s={summ['R^s']} sigma_cross={summ['sigma_cross']} R_min={summ['R_min']}")
        print(f"  macro: " + " ".join(f"{s}={v}" for s, v in summ["macro"].items()))
        print("  per-seq R@1 / sameplace / SpreadRate:")
        for k, v in per_seq.items():
            print(f"    {k:<20} {v['r1']:.3f}  {v['sameplace_med']:.3f}  {v['SpreadRate']:.0f}%")
    (REPO / "results/_metric_probe.json").write_text(json.dumps(report, indent=2))
    print("\nWROTE results/_metric_probe.json")


if __name__ == "__main__":
    main()
