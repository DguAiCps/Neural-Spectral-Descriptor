#!/usr/bin/env python3
"""검산 완료 비교 방법 설명을 읽기 순서에 맞는 작은 원고 조각으로 나눈다.

원래 세 생성 파일과 정적 진단 표는 보존한다. 읽기용 조각에서 현재 평가 조건과
대표 결과를 유지하며 중복과 공개 실험의 역추적 과정을 줄인다.
"""
import hashlib
import json
import re
from pathlib import Path


def _blocks(text: str) -> dict:
    """문단 제목 앞의 공통 머리말과 문단을 손실 없이 분리한다."""
    matches = list(re.finditer(r"\\paragraph\{([^}]*)\}", text))
    parts = {"_head": text[:matches[0].start()]}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        assert match.group(1) not in parts
        parts[match.group(1)] = text[match.start():end]
    assert "".join(parts.values()) == text
    return parts


def _body(block: str) -> str:
    return block.split("\n", 1)[1]


def _cut(text: str, anchors: list) -> list:
    """각 경계가 한 번만 나타나는지 확인하고 원문을 그대로 나눈다."""
    starts = [0]
    for anchor in anchors:
        assert text.count(anchor) == 1, anchor
        starts.append(text.index(anchor))
    assert starts == sorted(starts)
    pieces = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    assert "".join(pieces) == text
    return pieces


def _compact_native_protocols(fragments: dict) -> list:
    """현재 평가 조건을 남기고 공개 실험의 역추적 과정만 줄인다."""
    changes = []

    def replace_once(name: str, old: str, new: str, reason: str) -> None:
        assert fragments[name].count(old) == 1, (name, old)
        fragments[name] = fragments[name].replace(old, new)
        changes.append({
            "fragment": name, "original": old, "replacement": new,
            "reason": reason, "words_before": len(old.split()),
            "words_after": len(new.split()),
        })

    def replace_body(name: str, title: str, body: str, reason: str) -> None:
        old = _blocks(fragments[name])[title]
        new = "\\paragraph{" + title + "}\n" + body.strip() + "\n\n"
        replace_once(name, old, new, reason)

    scpp = _blocks(fragments["scpp_protocol"])
    replace_once("scpp_protocol", scpp["Published-curve checks."], "",
                 "벡터 PR 그림의 좌표·분모 역추정은 원천 기록에 보존한다.")
    replace_body(
        "scpp_protocol", "Revisit-list definition.",
        r"Under the paper's time-window revisit definition, with past nodes within 8\,m, no single time gap reproduces both the reported 852 KITTI~00 and 102 KITTI~08 revisits.",
        "서로 양립하지 않는 시간 간격의 세밀한 끝점 대신 재현 불일치 결론을 남긴다.",
    )
    replace_once(
        "scpp_protocol", " rather than the 4,043 that this model fixes for the PC curve", "",
        "삭제한 그림 역추정 문단에 의존하는 비교절을 함께 제거한다.",
    )
    replace_once(
        "scpp_protocol",
        "Agreement of the rounded PC value thus depends on the AUC convention, and this single run neither reproduces the KITTI 08 results nor identifies the authors' run.",
        "Agreement of the rounded PC value depends on the AUC convention.",
        "실제 수치와 모집단 차이가 이미 보여 주는 반복 한정을 줄인다.",
    )
    sampler_source = _blocks(fragments["bevplace_protocol"])["Released-run sampler."]
    assert "0.15211 lies 0.0004 below the three-seed range of the initial sampler (0.15252--0.15262)" in sampler_source
    assert "The pre-registered primary criterion is therefore not met, and its auxiliary criterion favors the initial sampler." in sampler_source
    replace_body(
        "bevplace_protocol", "Released-run sampler.",
        r"Table~\ref{tab:results} uses the current code's random-positive sampler. First-iteration losses recomputed from the released epoch-5 and epoch-10 states match the logged values within $10^{-3}$ only with the repository's initial hardest-positive sampler (commit 4ad02e31), but the logged epoch-mean loss lies 0.0004 below the range of three repeated runs with that sampler. The exact code files and full training procedure of that run remain unverified.",
        "RNG·재개 에폭·세부 손실값을 줄이고 현재 표본기, 초기 순전파 증거와 에폭 평균 불일치, 미확인 상태를 남긴다.",
    )
    replace_body(
        "ring_protocol", "RING evaluation scope.",
        r"""We follow the published T-RO version of RING++, with 20\,m map spacing, 5\,m query spacing, and a 10\,m revisit threshold. Its NCLT table specifies neither the session pair nor the evaluation region. We use the 2012-02-04 map and 2012-03-17 query sessions and the entire trajectory: the paper's resolution ablation reports the same 73.21\% for this pair, and its runtime analysis describes 253 map scans over the full trajectory. The official generator gives 254 map scans and 1,054 queries. Over all queries, RING reaches 62.52\% and exact-maximum RING++ with released normalization reaches 62.24\% (55.60\% with the released kernel in Table~\ref{tab:published_number_check}). Over the 992 queries with a map scan within 10\,m, RING and exact-maximum RING++ reach 66.43\% and 66.13\%. The paper defines the denominator only as total positives, so we report both. The published 0.1\,m voxel downsampling before neighbor search is omitted by the release; adding it once, fixed in advance, leaves both exact-maximum RING++ values unchanged.

Neither method uses learned weights. Our evaluation-set generator and NCLT loader match the release,\footnote{\href{https://github.com/lus6-Jenny/RING/blob/c79c58584ad841629506b5873ed1877c9c278ce5/evaluation/generate_evaluation_sets.py}{Official RING evaluation-set generator}.} and independent regeneration reproduces the ordered frame lists, positions, and positive counts. The authors' exact map/query lists and sampling phase, voxel implementation, and order of voxelization relative to ground removal and cropping remain unavailable. The NCLT loader keeps points more than 1\,m above the Segway wheel axle, 0.04\,m above the lidar; this removes the eight downward-looking beams in all 1,308 evaluated scans. The square $\pm$70\,m crop and a grid containing both 0 and $2\pi$ depart from the rotation-invariance assumptions. Adding only a 70\,m circular mask gives all-query R@1 of 63.19\% for RING and 62.62\% for exact-maximum RING++; the unmasked run reproduces the released distances bitwise. Excluding $2\pi$ gives 63.09\% and 61.95\%, respectively. These changes, single-map-scan removal, the released default test region or an alternate query session, and eigenentropy or local-feature coordinate changes do not close the gap to the published 70.98\% and 73.21\%. The Radon-grid, eigenentropy, and local-feature coordinate changes are not applied to the common-test correction. Rasterization and other untested effects remain unquantified.""",
        "대체 세션·스캔 제거의 개별 수치를 압축하고 현재 평가 설정·양쪽 분모·대표 결과·미확인 조건을 보존한다. 원고의 기존 RING 범위 문장에 있던 추가 진단의 미채택 사실을 이 조각에 한 번 둔다.",
    )
    changes[-1]["additional_source"] = {
        "path": "cvpr2027_supplement_body.tex (cleanup 이전 RING 범위 문단)",
        "original": "Separate changes to the Radon angle endpoints, eigenentropy convention, and local-feature coordinate scaling do not resolve the remaining discrepancy and are not applied to the common-test correction.",
    }
    replace_body(
        "disco_protocol", "Preprocessing variants.",
        r"With the same checkpoint, pairs, and labels, restoring the 5\,m ego-vehicle exclusion used at checkpoint upload, instead of the current 1\,m exclusion, gives R@1/R@1\% of 81.52/96.90\% versus 82.05/97.29\%. An earlier public generator rescaled each scan by its per-axis maxima and stored heights rather than binary occupancy. We use the fixed input matching the paper's definition; the training input of either checkpoint is unknown.",
        "현재·업로드 당시 전처리의 실제 결과는 유지하고 변경 commit 날짜만 줄인다.",
    )
    replace_body(
        "disco_protocol", "Evaluation-list sensitivity.",
        r"Keeping every second database sample changes mean R@1 from 82.05\% to 90.90/91.52\% for the two sampling phases and reduces queries with a positive from 3,173 to 1,715/1,689. With queries, weights, and the 1.5\,m radius fixed, replacing the 2012-03-17 database list with the first public sampler's list raises five-pair R@1 from 80.73\% to 89.45\% on the 427 queries positive under both lists. Both lists cover only the first 14\% of the session and differ in spacing and phase. These fixed-checkpoint diagnostics show sensitivity to list construction; they neither reproduce the published 30-pair value nor identify the authors' lists or the cause of the gap.",
        "목록 민감도의 대표 결과와 바뀌는 모집단을 남기고 이득 분해·중간 목록 크기·중복 평균을 줄인다.",
    )
    return changes


def write_baseline_reader_fragments(paper: Path, output: Path = None) -> dict:
    """기존 생성물과 원천 표를 읽고 방법별 조각 및 보존 목록을 쓴다."""
    output = output or paper / "baseline_fragments"
    output.mkdir(parents=True, exist_ok=True)
    originals = {
        name: (paper / name).read_text()
        for name in ("baseline_published_audit.tex", "baseline_common_diagnostics.tex",
                     "baseline_sampler_controls.tex", "tables_v3/appdx_baseline_diagnostics.tex")
    }
    published = _blocks(originals["baseline_published_audit.tex"])
    common = _blocks(originals["baseline_common_diagnostics.tex"])
    fragments = {}
    mapping = {}

    def emit(name: str, text: str, origin: str) -> None:
        assert name not in fragments
        fragments[name] = text
        mapping[name] = origin

    overview = published.pop("_head").replace(r"\subsection{Published-number and protocol audit}",
                                               r"\subsection{Released-code, released-weight, and published-protocol checks}")
    emit("published_overview", overview, "baseline_published_audit.tex: head and published table")
    for title, name in (
        ("SeqOT labels.", "seqot_distance_labels"),
        ("SeqOT overlap labels.", "seqot_overlap_labels"),
        ("RING evaluation scope.", "ring_protocol"),
        ("DiSCO released checkpoint.", "disco_protocol"),
        ("BEVPlace++ released checkpoint.", "bevplace_protocol"),
        ("Scan Context++ released code.", "scpp_protocol"),
        ("RING\\#-L released weights and recipe.", "ringsharp_protocol"),
        ("Common-task adaptations.", "common_task"),
    ):
        emit(name, published.pop(title), "baseline_published_audit.tex: " + title)
    sensor = _body(published.pop("Sensor and implementation adaptations."))
    titles = ("Sensor inputs.", "SeqOT input and tuple construction.", "HeLiOS runtime guards.",
              "Scan Context++ (A-PC).", "Scan Context++ height sensitivity.")
    names = ("sensor_inputs", "seqot_configuration", "helios_configuration", "scpp_configuration", "scpp_height")
    pieces = _cut(sensor, ["SeqOT keeps", "HeLiOS empty-miner", "Scan Context++ in Table", "A sensitivity run"])
    for title, name, piece in zip(titles, names, pieces):
        emit(name, "\\paragraph{" + title + "}\n" + piece.strip() + "\n\n",
             "baseline_published_audit.tex: Sensor and implementation adaptations.")
    assert not published

    head = common.pop("_head")
    head = head.replace(r"\subsection{Common-benchmark and training-signal diagnostics}" + "\n", "")
    head = head.replace(r"\label{app:baseline_signal}" + "\n", "")
    overview = head + common.pop("Retrieval and acceptance ranking.") + common.pop("Dataset dependence.")
    # RING++의 세 진단 행은 기존 변형 표와 동일하다. 새 원고는 해당 표를 참조한다.
    overview = overview.replace(r"\ref{tab:ringpp_maximum}", r"\ref{tab:baseline_release_comparison}")
    emit("common_metrics", overview, "baseline_common_diagnostics.tex: head, Retrieval and acceptance ranking., Dataset dependence.")
    for title, name in (
        ("Training targets.", "helios_targets"),
        ("SeqOT positive signal.", "seqot_signal"),
        ("Sensor and augmentation checks.", "seqot_sensor_checks"),
        ("Controlled yaw check.", "seqot_yaw"),
        ("Direction-dependent validation loss.", "learned_direction"),
    ):
        emit(name, common.pop(title), "baseline_common_diagnostics.tex: " + title)
    assert not common
    sampler = originals["baseline_sampler_controls.tex"].replace(
        r"\subsection{SeqOT positive-sampling controls}", r"\paragraph{Positive-sampling controls.}")
    emit("seqot_sampler", sampler, "baseline_sampler_controls.tex: entire file; heading depth only")

    # 원래 세 출력은 보존하고, 분할본에서 중복 안내와 가까운 문단의 자기 참조만 정리한다.
    reader_edits = (
        ("common_task", "The main results must be interpreted under this specified task. ", ""),
        ("seqot_signal", r" (Sec.~\ref{app:baselines})", ""),
    )
    for name, old, new in reader_edits:
        assert fragments[name].count(old) == 1, (name, old)
        fragments[name] = fragments[name].replace(old, new)

    # 긴 단락 안의 조건 전환에 제목을 붙인다. 제목 외 본문은 바꾸지 않는다.
    breaks = {
        "seqot_overlap_labels": [
            ("The released code applies these poses", "Overlap coordinate composition."),
            ("Translating the projection origin", "Overlap projection origin."),
        ],
        "disco_protocol": [
            ("The checkpoint is byte-identical", "Released-checkpoint identity."),
            ("With the same checkpoint, pairs, and labels", "Preprocessing variants."),
            ("Keeping every second database sample", "Evaluation-list sensitivity."),
        ],
        "bevplace_protocol": [
            ("The released training run is consistent", "Released-run sampler."),
            ("The paper states no positive-selection rule", "Paper and code recipes."),
        ],
        "scpp_protocol": [
            ("Under the paper's time-window revisit definition", "Revisit-list definition."),
            ("The PR curves in the authors' public arXiv source", "Published-curve checks."),
            ("Running the released PC and CC functions once", "All-frame released-code check."),
        ],
    }
    for name, boundaries in breaks.items():
        original = fragments[name]
        modified = original
        for anchor, heading in boundaries:
            assert modified.count(anchor) == 1, (name, anchor)
            modified = modified.replace(anchor, "\n\n\\paragraph{" + heading + "}\n" + anchor)
        # 새 제목과 공백을 빼면 원문과 정확히 같다.
        restored = modified
        for _, heading in boundaries:
            restored = restored.replace("\n\n\\paragraph{" + heading + "}\n", "")
        assert restored == original
        fragments[name] = modified

    reader_compactions = _compact_native_protocols(fragments)

    diagnostic_tables = originals["tables_v3/appdx_baseline_diagnostics.tex"]
    native_start = diagnostic_tables.index(r"\begin{table*}")
    native = diagnostic_tables[native_start:]
    assert r"\label{tab:baseline_native}" in native and r"\label{tab:ringpp_maximum}" not in native
    emit("native_table", native, "tables_v3/appdx_baseline_diagnostics.tex: native-protocol table")
    # 공통 시험 RING++ 세 행은 기존 variant 표에 모두 있다. 원천 표는 지우지 않는다.
    variant = (paper / "tables_v3/appdx_baseline_variants.tex").read_text()
    triples = [
        ("RING++ released kernel", "Released kernel + released normalization"),
        ("RING++ exact max", "Exact max + released normalization"),
        (r"\quad + unit variance", r"Table~\ref{tab:results}: exact max + unit variance"),
    ]
    for left, right in triples:
        a = next(line for line in diagnostic_tables.splitlines() if line.startswith(left + " &"))
        b = next(line for line in variant.splitlines() if " & " + right + " & " in line)
        assert a.split(" & ")[-4:] == b.split(" & ")[-4:], (left, right)
    ring = next(line for line in diagnostic_tables.splitlines() if line.startswith("RING &"))
    assert ring == r"RING & 63.15 & 65.86 & 23.66 & 77.51 \\", ring

    manifest = {
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in originals.items()},
        "variant_table_sha256": hashlib.sha256(variant.encode()).hexdigest(),
        "reader_edits": [{"fragment": name, "removed": old, "replacement": new}
                         for name, old, new in reader_edits],
        "reader_compactions": reader_compactions,
        "fragments": {},
        "duplicate_ringpp_rows": "all three metric vectors equal appdx_baseline_variants.tex",
        "ring_reference_row": ring,
    }
    for name, text in fragments.items():
        # 분할된 파일만 열어도 원래 결과 원천을 찾을 수 있게 주석을 함께 둔다.
        origin = mapping[name].split(":", 1)[0]
        if origin in originals and not text.startswith("% 원천:"):
            provenance = originals[origin].split(r"\subsection", 1)[0]
            if origin.startswith("tables_v3/"):
                provenance = originals[origin].split(r"\begin{table}", 1)[0]
            text = provenance + text
        (output / (name + ".tex")).write_text(text)
        manifest["fragments"][name + ".tex"] = {
            "source": mapping[name], "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
