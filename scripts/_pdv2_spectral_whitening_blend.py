#!/usr/bin/env python3
"""상관 보정의 계열 손실을 확인한 뒤 고정된 두 혼합률을 비교한다."""
from pathlib import Path

import numpy as np

import _pdv2_spectral_whitening as source
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json

run = source.prior
OUT = source.OUT / "blend"
ARMS = {"baseline": 0.0, "quarter": 0.25, "half": 0.5, "full": 1.0}


def main() -> None:
    """후속 검증 탐색임을 명시하고 두 혼합률의 최종 손익을 검산한다."""
    old = source.protocol()
    OUT.mkdir(exist_ok=True)
    paths = {
        Path(__file__),
        source.OUT / "protocol.json",
        source.OUT / "summary.json",
        source.OUT / "completion_validation.json",
    }
    paths.update(
        (source.OUT / f"s{s}" / (q.replace("/", "_") + ".npz"))
        for s in run.SEEDS
        for q in old["val"]
    )
    spec = {
        "arms": ARMS,
        "test_used": False,
        "reason": "전체상관보정3시드평균AP+1.3188%p·R100+11.0073%p이지만HeLiPR AP-2.1176%p여서기존점수의절반또는3/4을보존하는두조건을후속고정한다. 첫결과를본뒤추가한검증탐색이며독립확증이아니다.",
        "formula": "score=(1-lambda)*original+lambda*full_whitened,lambda=.25/.5. 두점수는동일c_col/c_spec및동일최종시간판정단위. 승자/표본/학습통계불변.",
        "selection": "평균AP+.1%p이상·모든시드AP개선·평균R100손실.1%p이하·계열AP손실.25%p이하를모두통과한새조건중평균AP최대. 동률은작은혼합률. 실패하면기존유지. 이실행은시험/주표교체없음.",
        "sources": {str(p.relative_to(run.REPO)): sha256(p) for p in sorted(paths)},
    }
    protocol = OUT / "protocol.json"
    if protocol.exists():
        assert run.read(protocol) == spec
    else:
        write_json(protocol, spec)
    per_seed, stats, checks = {}, {}, []
    for arm, fraction in ARMS.items():
        per_seed[arm] = {}
        for seed in run.SEEDS:
            rows = {}
            for sequence in old["val"]:
                path = source.OUT / f"s{seed}" / (sequence.replace("/", "_") + ".npz")
                row = run.BASE.exp.arrays(path)
                value = (1 - fraction) * row["identity"] + fraction * row["full"]
                assert np.isfinite(value).all()
                rows[sequence] = independent_metrics(row, value)
                other = run.BASE.exp.metrics(value, row)
                for key in run.METRICS:
                    if rows[sequence][key] is not None:
                        np.testing.assert_allclose(
                            rows[sequence][key], other[key], atol=1e-12, rtol=0
                        )
                folder = OUT / arm / f"s{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                output = folder / path.name
                np.savez_compressed(
                    output, **{k: row[k] for k in run.BASE.FIELDS}, score=value
                )
                checks.append(
                    {
                        "path": str(output.relative_to(run.REPO)),
                        "sha256": sha256(output),
                        "rows": len(value),
                    }
                )
            per_seed[arm][str(seed)] = {**run.BASE.macro(rows), "sequences": rows}
        stats[arm] = {
            m: {
                "mean": float(np.mean([v["macro"][m] for v in per_seed[arm].values()])),
                "std": float(
                    np.std([v["macro"][m] for v in per_seed[arm].values()], ddof=1)
                ),
            }
            for m in run.METRICS
        }
    original = run.read(source.OUT / "summary.json")
    for a, b in (("baseline", "identity"), ("full", "full")):
        assert stats[a] == original["stats"][b]
    decisions = {}
    for arm in ("quarter", "half"):
        delta = {
            m: 100 * (stats[arm][m]["mean"] - stats["baseline"][m]["mean"])
            for m in run.METRICS
        }
        per_seed_delta = {
            str(s): 100
            * (
                per_seed[arm][str(s)]["macro"]["AP"]
                - per_seed["baseline"][str(s)]["macro"]["AP"]
            )
            for s in run.SEEDS
        }
        family = {
            f: 100
            * float(
                np.mean(
                    [
                        per_seed[arm][str(s)]["families"][f]["AP"]
                        - per_seed["baseline"][str(s)]["families"][f]["AP"]
                        for s in run.SEEDS
                    ]
                )
            )
            for f in ("kitti", "helipr", "mulran")
        }
        passed = {
            "mean_AP": delta["AP"] >= 0.1,
            "all_seeds_AP": min(per_seed_delta.values()) > 0,
            "R100": delta["R@100P"] >= -0.1,
            "family_AP": min(family.values()) >= -0.25,
        }
        decisions[arm] = {
            "delta_pp": delta,
            "seed_AP_delta_pp": per_seed_delta,
            "family_AP_delta_pp": family,
            "checks": passed,
            "promoted": all(passed.values()),
        }
    eligible = [a for a, d in decisions.items() if d["promoted"]]
    selected = (
        max(eligible, key=lambda a: (stats[a]["AP"]["mean"], -ARMS[a]))
        if eligible
        else "baseline"
    )
    write_json(
        OUT / "summary.json",
        {
            "per_seed": per_seed,
            "stats": stats,
            "decisions": decisions,
            "selected": selected,
            "selection_split": "validation",
            "independent_confirmation": False,
            "test_used": False,
        },
    )
    write_json(
        OUT / "completion_validation.json",
        {
            "status": "passed",
            "ledgers": len(checks),
            "query_rows": sum(r["rows"] for r in checks),
            "checks": checks,
            "endpoint_reproduction_exact": True,
            "prefix_property": "같은원장의기존/보정점수각각에서검사한q+15접두동일성이선형혼합에서도성립한다.",
            "protocol_sha256": sha256(protocol),
            "summary_sha256": sha256(OUT / "summary.json"),
            "test_used": False,
        },
    )
    print("FINAL_BLEND", stats, decisions, "SELECTED", selected, flush=True)


if __name__ == "__main__":
    main()
