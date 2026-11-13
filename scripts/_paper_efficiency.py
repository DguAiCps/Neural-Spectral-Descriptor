"""검증된 현재 주모델의 효율 원천을 원고 표에 연결한다."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = "results/pdv2_table3_20261029/summary.json"
METHODS = (("disco", "DiSCO"), ("deepring", "DeepRING"), ("scpp", "SC++"),
           ("bevplace", "BEVPlace++"), ("ring", "RING"), ("ringpp", "RING++"),
           ("ringsharp", r"RING\#-L"), ("seqot", "SeqOT"), ("helios", "HeLiOS"),
           ("imlpr", "ImLPR"), ("nsd", r"\textbf{NSD}"))
DASH = r"\cmedit{---}"
# 표 3의 최초 후보 수. 인식 평가의 Sejong01 갤러리(약 18,400 키프레임)에 가까운 크기로 정했다(2026-10-31 저자 결정).
TABLE3_N = 10000
OTHER_N = tuple(n for n in (1000, 5000, 10000) if n != TABLE3_N)


def apply_efficiency(tables: dict, manifest: dict) -> None:
    """완료·검증·해시 조건을 통과한 측정값만 생성한다."""
    source = REPO / SOURCE
    rows = []
    if source.exists():
        value = json.loads(source.read_text())
        rows = [r for r in value["measurements"] if r["status"] == "passed" and r.get("publish_ready", True)]
        for row in rows:
            assert row["verification"]["status"] == "passed"
            assert row["protocol_sha256"] == value["protocol_sha256"]
            path = REPO / "results/pdv2_table3_20261029/methods" / f"{row['method']}_n{row['n_map']}.json"
            actual = json.loads(path.read_text())
            for field in ("timings_ms", "resident_bytes", "verification"):
                assert row[field] == actual[field]
        manifest["sources"].append({"path": SOURCE, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size})
        manifest["source_paths"].append(SOURCE)
        provenance = REPO / 'results/pdv2_table3_20261029/cache_provenance_manifest.json'
        if provenance.exists():
            name = str(provenance.relative_to(REPO))
            manifest["sources"].append({"path": name, "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(), "bytes": provenance.stat().st_size})
            manifest["source_paths"].append(name)
    # 완료된 새 주모델의 실제 측정만 교체한다. 동결된 비교 방법 원천은 보존한다.
    power_root = REPO / "results/pdv2_spectral_power_20261101/adoption/table3"
    power_rows, power_sources = [], []
    for size in (1000, 5000, 10000):
        path = power_root / f"nsd_n{size}.json"
        row = json.loads(path.read_text())
        assert row["status"] == "passed" and row["publish_ready"]
        assert row["method_protocol_sha256"] == hashlib.sha256((power_root / "protocol.json").read_bytes()).hexdigest()
        assert row["source_sha256"] == hashlib.sha256((REPO / "scripts/_pdv2_power_adoption_benchmark.py").read_bytes()).hexdigest()
        assert row["candidate_range"] == [size, size + 95] and len(row["raw_samples"]) == 288
        assert row["verification"]["new_final_confidence"]["queries"] == 288
        assert row["verification"]["new_final_confidence"]["winners_exact"]
        assert row["verification"]["new_final_confidence"]["score_max_abs_error"] <= 1e-6
        power_rows.append(row)
        relative = str(path.relative_to(REPO))
        power_sources.append(relative)
        manifest["sources"].append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
        manifest["source_paths"].append(relative)
    rows = [r for r in rows if r["method"] != "nsd"] + power_rows
    selected = {r["method"]: r for r in rows if r["n_map"] == TABLE3_N}
    text = ["% scripts/_paper_efficiency.py가 검증된 효율 원천으로 생성한다.",
        r"\begin{table}[!htbp]",
        r"\caption{\cmedit{\textbf{Storage and computation.} MulRan Sejong01, RTX~5080, one CPU thread, batch~1; seed~1 for learned methods. The eligible gallery grows from %s to %s frames over 96 arrivals (three repeats). Resident state counts unique CPU buffers and persistent GPU allocations, including model, context and workspace, at the final step, divided by %s. Times are means; update includes raw encoding and actual state/index update, query includes final confidence. Shared odometry and disk I/O are excluded. Dashes indicate unmeasured or unresolved costs.}}" % ("{:,}".format(TABLE3_N), "{:,}".format(TABLE3_N+95), "{:,}".format(TABLE3_N+95)),
        r"\label{tab:efficiency}", r"\centering\footnotesize\color{blue}",
        r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lccc}", r"\toprule",
        r"Method & \shortstack{Resident state\\(KB/frame)} & \shortstack{Actual update\\(ms/keyframe)} & \shortstack{Query\\(ms)} \\", r"\midrule"]
    for key, name in METHODS:
        row = selected.get(key)
        if row is None:
            cells = [DASH]*3
        else:
            total = row["resident_bytes"].get("total")
            memory_ready = row.get("memory_publish_ready", True) and row["resident_bytes"].get("publish_ready", True) and total is not None
            cells = [r"\cmedit{%.2f}" % (total/row["candidate_range"][1]/1000) if memory_ready else DASH]
            cells += [r"\cmedit{%.2f}" % row["timings_ms"][k]["mean"] for k in ("update", "query")]
        text.append(name + " & " + " & ".join(cells) + r" \\")
    text += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tables["table3_efficiency.tex"] = "\n".join(text)
    detail = ["% scripts/_paper_efficiency.py가 검증된 효율 원천으로 생성한다.",
        r"\begin{table*}[t]", r"\centering\footnotesize\color{blue}",
        r"\caption{\cmedit{\textbf{Measured resource accounting.} The first eligible gallery size is $N$ and the final size is $N+95$; 45 observed support frames are additionally retained. Decimal MB and milliseconds. Resident state counts unique CPU buffers, PyTorch GPU allocations and separately traced persistent native buffers once, after the third repetition. Native bytes denote requested capacity, not driver allocation granularity. RSS also includes the interpreter, libraries and preloaded input/audit fixtures. Peak GPU is the PyTorch allocator peak and excludes temporary native allocations. Unresolved native allocations leave the resident entry blank. Timing cells show mean/p95 over 96 queries, repeated three times. Cached initial-map construction is excluded from actual updates. NSD recomputes the last 11 keys using a 31-frame local graph.}}",
        r"\label{tab:cost_accounting}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrllllll}",
        r"\toprule", r"Method & $N$ & Resident MB & RSS MB & Peak GPU MB & Update ms & Query ms & Processing ms \\", r"\midrule"]
    labels = dict(METHODS)
    for row in sorted(rows, key=lambda r: (r["n_map"], r["method"])):
        memory = row["resident_bytes"]
        cells = [labels[row["method"]], str(row["n_map"]), "---" if memory.get("total") is None or not row.get("memory_publish_ready", True) else "%.2f" % (memory["total"]/1e6),
                 "%.1f" % (memory["process_rss_after"]/1e6), "%.2f" % (memory["cuda_peak_allocated"]/1e6)]
        cells += ["%.2f/%.2f" % (row["timings_ms"][k]["mean"], row["timings_ms"][k]["p95"]) for k in ("update", "query", "processing")]
        detail.append(" & ".join(cells) + r" \\")
    if not rows:
        detail.append("Pending & --- & --- & --- & --- & --- & --- & --- " + r"\\")
    detail += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    tables["table_cost_accounting.tex"] = "\n".join(detail)
    scale = [r"\begin{table*}[t]", r"\centering\footnotesize\color{blue}",
        r"\caption{\cmedit{\textbf{Processing cost across map sizes.} Same workload and accounting as Table~\ref{tab:efficiency}. Each timing is mean/p95 (ms). State update excludes raw encoding, whereas total processing includes encoding, state update, retrieval/reranking and final confidence. The 15-keyframe observation delay is not a computation time.}}",
        r"\label{tab:e2e_latency}", r"\begin{tabular}{lrllll}", r"\toprule",
        r"Method & Initial $N$ & Raw encoding & State update & Query & Total processing \\", r"\midrule"]
    indexed = {(r["method"], r["n_map"]): r for r in rows}
    for n_map in (1000, 5000, 10000):
        for method in ("nsd", "bevplace"):
            row = indexed.get((method, n_map))
            cells = ["---"]*4 if row is None else ["%.2f/%.2f" % (row["timings_ms"][k]["mean"], row["timings_ms"][k]["p95"]) for k in ("encode", "state_update", "query", "processing")]
            scale.append(labels[method] + " & " + str(n_map) + " & " + " & ".join(cells) + r" \\")
    scale += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    tables["table_scale_latency.tex"] = "\n".join(scale)
    if "nsd" in selected and "bevplace" in selected:
        nsd, bev = selected["nsd"], selected["bevplace"]
        nt, bt = nsd["timings_ms"]["processing"]["mean"], bev["timings_ms"]["processing"]["mean"]
        nm, bm = nsd["resident_bytes"]["total"]/1e6, bev["resident_bytes"]["total"]/1e6
        nq, bq = nsd["timings_ms"]["query"]["mean"], bev["timings_ms"]["query"]["mean"]
        query = (r"NSD is also faster in query alone (%.2f versus %.2f\,ms)." % (nq, bq) if nq < bq else
                 r"Query alone is faster for BEVPlace++ (%.3f versus %.3f\,ms); NSD's complete-processing advantage comes from encoding and state update." % (bq, nq))
        paragraph = (r"\cmedit{For the measured Sejong01 workload with an initial eligible gallery of " + "{:,}".format(TABLE3_N) + r" frames, "
            r"NSD requires %.2f\,ms per complete processing step versus %.2f\,ms for BEVPlace++ (%.2f$\times$ faster), "
            r"with %.2f versus %.2f\,MB of resident state (%.1f\%% less). "
            r"These are seed-1 resource measurements on the specified workload, not a claim of this speedup on every dataset. "
            r"The resident-state measure includes GPU allocated workspace and is distinct from process RSS. "
            r"Both methods use the same precision policy; BEVPlace++ retains official exact L2 retrieval with an AVX512-optimized FAISS CPU build. "
            r"%s}" % (nt, bt, bt/nt, nm, bm, 100*(1-nm/bm), query))
    else:
        paragraph = r"\cmedit{Matched measurements for the current implementation are being verified; only completed entries are reported in Table~\ref{tab:efficiency}.}"
    if (len(selected) == len(METHODS)
            and all(r["resident_bytes"].get("total") is not None and r.get("memory_publish_ready", True) for r in selected.values())
            and min(selected, key=lambda m: selected[m]["resident_bytes"]["total"]) == "nsd"):
        fastest = min(selected, key=lambda m: selected[m]["timings_ms"]["processing"]["mean"])
        paragraph += ("\n" + r"\cmedit{At this initial map size, NSD has the smallest measured resident state among the eleven methods. "
            r"%s has the lowest complete-processing time (%.2f\,ms); NSD's efficiency advantage is therefore a resource--accuracy tradeoff, not universal runtime dominance.}" % (
                dict(METHODS)[fastest], selected[fastest]["timings_ms"]["processing"]["mean"]))
    if all((method, size) in indexed for method in ("nsd", "bevplace") for size in OTHER_N):
        query_times = [indexed[(method, size)]["timings_ms"]["query"]["mean"]
                       for method in ("nsd", "bevplace") for size in OTHER_N]
        paragraph += ("\n" + r"\cmedit{At initial galleries of %s frames, query times are %%.2f/%%.2f\,ms for NSD and %%.2f/%%.2f\,ms for BEVPlace++ (detailed in the supplement).}" % "/".join("{:,}".format(n) for n in OTHER_N) % tuple(query_times))
    tables["efficiency_summary.tex"] = "% 검증된 원천에서 자동 생성한다.\n" + paragraph + "\n"
    environment_path = REPO / "results/pdv2_table3_20261029/environment.json"
    environment_text = r"\cmedit{Hardware and library details are recorded in the measurement manifest.}"
    if environment_path.exists():
        env = json.loads(environment_path.read_text())
        cpu = next(item["data"] for item in env["lscpu"]["lscpu"] if item["field"] == "Model name:")
        environment_text = (r"\cmedit{The CPU is %s; the RTX~5080 uses driver 610.88. "
            r"GPU methods run in the same container under WSL2 with PyTorch 2.10.0a0 (NVIDIA 26.01) and CUDA 13.1. "
            r"SC++ uses host MATLAB R2026a, Update~5, with one computational thread. Hardware clocks use the default policy. "
            r"FAISS 1.11.0 uses its AVX512 CPU build and exact IndexFlatL2. Against the same-version generic build, "
            r"all 96 BEVPlace++ winners agree, with maximum L2-score difference $2.07\times10^{-6}$. "
            r"SeqOT processes every newly observed raw scan between keyframe arrivals and reuses overlapping range/feature windows. "
            r"The initial-$N=1{,}000$ workload contains 220 new raw scans, whereas the $5{,}000$ and $10{,}000$ workloads each contain 96; "
            r"encoding-time changes therefore also reflect input density. "
            r"DeepRING uses the same frozen paper-based reimplementation as the recognition comparison. "
            r"ImLPR and HeLiOS retain the recognition evaluation's exact prefix cosine search; DiSCO retains its float64 prefix Euclidean search and exact shortlist rescoring. "
            r"Official KDTree top-1 queries are untimed correctness references for these three methods, not the timed retrieval backend.}" % cpu)
        manifest["sources"].append({"path": str(environment_path.relative_to(REPO)), "sha256": hashlib.sha256(environment_path.read_bytes()).hexdigest(), "bytes": environment_path.stat().st_size})
        manifest["source_paths"].append(str(environment_path.relative_to(REPO)))
    # 아래 진단 문장은 최초 N=1,000 실행을 설명한다.
    diagnostic = {r["method"]: r for r in rows if r["n_map"] == 1000}
    ringsharp = diagnostic.get("ringsharp")
    if ringsharp is not None and ringsharp.get("method_config", {}).get("angle_cnn_batch") == 16:
        environment_text += ("\n\n" + r"\cmedit{RING\#-L processes the unique rotated query features in batches of 16 to bound CNN workspace. "
            r"Exhaustive rotation retrieval, the top-1,000 translation candidates and the 128-candidate correlation batches are unchanged. "
            r"All 288 timed-query winners agree with the original implementation, with all candidate translation scores checked after timing. "
            r"The number of unique query rotations also varies by scene, so its query cost need not increase monotonically across these different arrival segments.}")
    ringpp = diagnostic.get("ringpp")
    if ringpp is not None:
        variation = ringpp["verification"]["official_native_repeat_variation"]
        environment_text += ("\n\n" + r"\cmedit{RING++ computes exact candidate correlations in chunks of 128 to bound temporary GPU workspace. "
            r"The scores and selected places agree with the official exhaustive scorer. "
            r"RING++ preserves the official non-atomic native feature maximum. "
            r"At initial $N=1{,}000$, independent forward passes changed the winner for %d of 96 queries across three repetitions; "
            r"the maximum final-confidence spread was %.5f. Cost validation instead compares each actual execution "
            r"with the official exhaustive scorer: all 288 winners agree. The native feature routine retains 360 requested "
            r"CUDA bytes per encoding. We count 108,000 bytes after 12 warm-up and 288 measured encodings in a new process "
            r"initialized from cached descriptors; the separate cache-building process is no longer live. "
            r"This snapshot does not represent the leak accumulated over a longer deployment. "
            r"Native allocation granularity and temporary native peaks are outside the reported PyTorch peak.}" % (
                variation["winner_changed_queries"], variation["max_confidence_spread"]))
    helios = diagnostic.get("helios")
    if helios is not None:
        environment_text += ("\n\n" + r"\cmedit{HeLiOS shows small FP32 inference variation even for an identical GPU input batch. "
            r"Its maximum cached-versus-recomputed confidence difference is $1.20\times10^{-7}$ in the initial-$N=1{,}000$ diagnostic. "
            r"All 288 winners agree with the reference cache; same-execution exhaustive search, KDTree top-1 and final dot scores also agree. "
            r"Feature perturbation bounds are smaller than the corresponding top-1/top-2 margins. "
            r"The official model, preprocessing and precision are unchanged, as is the frozen recognition evaluation's exact search computation.}")
    tables["efficiency_environment.tex"] = environment_text + "\n"
    manifest["tables"]["efficiency_environment.tex"] = {"source_paths": [str(environment_path.relative_to(REPO))], "role": "measured_hardware_and_protocol"}
    manifest["output_sha256"]["efficiency_environment.tex"] = hashlib.sha256(tables["efficiency_environment.tex"].encode()).hexdigest()
    for name, label in (("table3_efficiency.tex", "tab:efficiency"), ("table_cost_accounting.tex", "tab:cost_accounting"), ("table_scale_latency.tex", "tab:e2e_latency")):
        manifest["tables"][name] = {"label": label, "source_paths": ([SOURCE] if source.exists() else []) + power_sources,
            "status": "measured_partial" if rows else "pending", "measured": [{"method": r["method"], "n_map": r["n_map"]} for r in rows]}
        manifest["output_sha256"][name] = hashlib.sha256(tables[name].encode()).hexdigest()
    manifest["tables"]["efficiency_summary.tex"] = {"source_paths": [SOURCE] + power_sources, "role": "validated_efficiency_prose"}
    manifest["output_sha256"]["efficiency_summary.tex"] = hashlib.sha256(tables["efficiency_summary.tex"].encode()).hexdigest()
