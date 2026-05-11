#!/usr/bin/env python3
"""Train a lightweight learned phase-sketch reranker on KITTI.

This script freezes the NSD encoder/GNN retrieval key and trains only a small
candidate-level reranker over top-N candidates. The reranker sees:

* frozen embedding cosine similarity
* the full cyclic phase-correlation curve from compact BEV phase coefficients

It is a two-stage method by design. Do not merge its numbers into the pure
single-pass NSD/GNN row.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gnn.learned_reranker import PhaseCorrelationReranker  # noqa: E402
from run_kitti_bev_layout_rerank import _build_bev_layout_cache, _pool_rows  # noqa: E402
from run_kitti_operating_point import _find_queries, _normalize, _score, _topk_cosine  # noqa: E402
from evaluate_kitti_checkpoint import (  # noqa: E402
    _apply_encoder_preset,
    _build_eval_graph,
    _build_sequence_cache,
    _cache_to_keyframes,
    _load_config,
    _make_model,
    _phase_sketch,
    _phase_sketch_keys,
)


def _load_embeddings(
    cache: np.lib.npyio.NpzFile,
    config: Dict,
    checkpoint: Path | None,
    device: str,
    temporal_edge_mode: str,
    temporal_direction_mode: str,
    similarity_min_k: int,
    sensor_key: str = "kitti",
) -> np.ndarray:
    descriptors = cache["descriptors"].astype(np.float32)
    if checkpoint is None:
        return descriptors

    keyframes = _cache_to_keyframes(cache)
    poses = cache["poses"]
    model = _make_model(config, checkpoint, device)
    graph = _build_eval_graph(
        keyframes=keyframes,
        poses=poses,
        descriptors=descriptors,
        cache=cache,
        config=config,
        device=device,
        temporal_edge_mode=temporal_edge_mode,
        temporal_direction_mode=temporal_direction_mode,
        similarity_min_k=similarity_min_k,
        phase_features=None,
        sensor_key=sensor_key,
    )
    with torch.no_grad():
        embeddings = model(graph.to(device)).detach().cpu().numpy().astype(np.float32)
    return embeddings


def _prepare_sequence(
    sequence: str,
    root: Path,
    config: Dict,
    checkpoint: Path | None,
    cache_dir: Path,
    bev_cache_dir: Path,
    device: str,
    layout_sectors: int,
    bev_freqs: int,
    bev_row_pool: int,
    bev_row_pool_mode: str,
    temporal_edge_mode: str,
    temporal_direction_mode: str,
    similarity_min_k: int,
    bev_min_range: float,
    bev_max_range: float,
    bev_z_min: float,
    bev_z_max: float,
    bev_height_layers: int,
    bev_height_encoding: str,
) -> Dict:
    cache_path = _build_sequence_cache(
        root=root,
        sequence=sequence,
        config=config,
        cache_dir=cache_dir,
        device=device,
        layout_sectors=layout_sectors,
    )
    cache = np.load(cache_path)
    bev_path = bev_cache_dir / (
        f"kitti_bev_layout_{sequence}_s{layout_sectors}_"
        f"{bev_height_encoding}_r{bev_min_range:g}-{bev_max_range:g}_"
        f"z{bev_z_min:g}-{bev_z_max:g}_h{bev_height_layers}.npz"
    )
    bev_layouts = _build_bev_layout_cache(
        root=root,
        sequence=sequence,
        base_cache=cache,
        output_path=bev_path,
        n_sectors=layout_sectors,
        max_range=bev_max_range,
        min_range=bev_min_range,
        z_min=bev_z_min,
        z_max=bev_z_max,
        n_height_layers=bev_height_layers,
        height_encoding=bev_height_encoding,
    )
    bev_layouts = _pool_rows(
        bev_layouts,
        bev_row_pool,
        bev_row_pool_mode,
        n_channels=3 if bev_height_encoding == "physics3" else 1,
    )
    embeddings = _load_embeddings(
        cache=cache,
        config=config,
        checkpoint=checkpoint,
        device=device,
        temporal_edge_mode=temporal_edge_mode,
        temporal_direction_mode=temporal_direction_mode,
        similarity_min_k=similarity_min_k,
    )
    sketch = _phase_sketch(bev_layouts.astype(np.float32), bev_freqs)
    return {
        "sequence": sequence,
        "poses": cache["poses"].astype(np.float64),
        "embeddings": embeddings.astype(np.float32),
        "emb_norm": _normalize(embeddings),
        "phase_sketch": sketch,
        "phase_keys": _phase_sketch_keys(sketch),
    }


def _candidate_set(
    seq: Dict,
    query_idx: int,
    n_coarse: int,
    max_candidates: int,
    skip_frames: int,
    include_phase_candidates: bool,
) -> np.ndarray:
    cand_parts = [
        _topk_cosine(seq["emb_norm"], query_idx, n_coarse, skip_frames),
    ]
    if include_phase_candidates:
        cand_parts.append(_topk_cosine(seq["phase_keys"], query_idx, n_coarse, skip_frames))
    seen = set()
    ordered: List[int] = []
    for cand in np.concatenate(cand_parts):
        c = int(cand)
        if c == query_idx or c in seen:
            continue
        seen.add(c)
        ordered.append(c)
        if len(ordered) >= max_candidates:
            break
    return np.asarray(ordered, dtype=np.int64)


def _phase_corr_batch(
    sketch: np.ndarray,
    query_indices: np.ndarray,
    candidate_indices: np.ndarray,
    valid_mask: np.ndarray,
    n_sectors: int,
) -> np.ndarray:
    # sketch: (M,R,K) complex. candidate_indices: (B,N), padded with 0.
    q = sketch[query_indices]                                      # (B,R,K)
    c = sketch[np.maximum(candidate_indices, 0)]                    # (B,N,R,K)
    cross = np.einsum("brk,bnrk->bnk", np.conj(q), c, optimize=True)
    B, N, K = cross.shape
    padded = np.zeros((B, N, n_sectors), dtype=np.complex64)
    padded[:, :, 1 : K + 1] = cross.astype(np.complex64)
    corr = np.fft.fft(padded, axis=-1).real.astype(np.float32)

    q_norm = np.linalg.norm(q.reshape(B, -1), axis=1)              # (B,)
    c_norm = np.linalg.norm(c.reshape(B, N, -1), axis=2)           # (B,N)
    denom = np.maximum(q_norm[:, None] * c_norm, 1e-8)
    corr = corr / denom[:, :, None]
    corr[~valid_mask] = 0.0
    return corr.astype(np.float32)


def _make_batch(
    seqs: List[Dict],
    query_refs: List[tuple[int, int]],
    batch_refs: List[tuple[int, int]],
    n_coarse: int,
    max_candidates: int,
    skip_frames: int,
    distance_threshold: float,
    include_phase_candidates: bool,
    device: str,
    n_sectors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(batch_refs)
    cand = np.full((B, max_candidates), -1, dtype=np.int64)
    queries = np.empty((B,), dtype=np.int64)
    seq_ids = np.empty((B,), dtype=np.int64)
    valid = np.zeros((B, max_candidates), dtype=bool)
    labels = np.zeros((B, max_candidates), dtype=bool)
    emb_sim = np.zeros((B, max_candidates), dtype=np.float32)

    for b, (seq_id, q_idx) in enumerate(batch_refs):
        seq = seqs[seq_id]
        cands = _candidate_set(
            seq,
            q_idx,
            n_coarse=n_coarse,
            max_candidates=max_candidates,
            skip_frames=skip_frames,
            include_phase_candidates=include_phase_candidates,
        )
        n = min(len(cands), max_candidates)
        if n == 0:
            continue
        cand[b, :n] = cands[:n]
        queries[b] = q_idx
        seq_ids[b] = seq_id
        valid[b, :n] = True
        dxy = seq["poses"][cands[:n], :2, 3] - seq["poses"][q_idx, :2, 3][None, :]
        labels[b, :n] = np.linalg.norm(dxy, axis=1) <= distance_threshold
        emb_sim[b, :n] = seq["emb_norm"][cands[:n]] @ seq["emb_norm"][q_idx]

    # This helper assumes all rows in a batch come from the same sequence for
    # fast vectorized phase correlation. The caller enforces that.
    if len(set(seq_ids.tolist())) != 1:
        raise ValueError("batch must contain queries from one sequence")
    sketch = seqs[int(seq_ids[0])]["phase_sketch"]
    shift_corr = _phase_corr_batch(sketch, queries, cand, valid, n_sectors=n_sectors)
    return (
        torch.from_numpy(shift_corr).to(device),
        torch.from_numpy(emb_sim).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(labels).to(device),
    )


def _listwise_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pos = labels & valid
    keep = pos.any(dim=1)
    if not keep.any():
        return logits.sum() * 0.0
    logits = logits[keep]
    pos = pos[keep]
    target = pos.float() / pos.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _query_refs(seqs: List[Dict], distance_threshold: float, skip_frames: int) -> List[tuple[int, int]]:
    refs: List[tuple[int, int]] = []
    for sid, seq in enumerate(seqs):
        for q_idx, _ in _find_queries(seq["poses"], distance_threshold, skip_frames):
            refs.append((sid, int(q_idx)))
    return refs


def _evaluate(
    model: PhaseCorrelationReranker,
    seqs: List[Dict],
    refs: List[tuple[int, int]],
    n_coarse: int,
    max_candidates: int,
    skip_frames: int,
    distance_threshold: float,
    include_phase_candidates: bool,
    device: str,
    n_sectors: int,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    by_seq: Dict[int, List[int]] = {}
    for sid, q in refs:
        by_seq.setdefault(sid, []).append(q)
    out: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for sid, q_list in by_seq.items():
            seq = seqs[sid]
            ranked_lists: List[np.ndarray] = []
            for start in range(0, len(q_list), 16):
                batch_q = q_list[start : start + 16]
                batch_refs = [(sid, q) for q in batch_q]
                shift_corr, emb_sim, valid, _ = _make_batch(
                    seqs,
                    refs,
                    batch_refs,
                    n_coarse,
                    max_candidates,
                    skip_frames,
                    distance_threshold,
                    include_phase_candidates,
                    device,
                    n_sectors,
                )
                logits = model(shift_corr, emb_sim, valid)
                order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
                for row, q_idx in enumerate(batch_q):
                    cands = _candidate_set(
                        seq,
                        q_idx,
                        n_coarse,
                        max_candidates,
                        skip_frames,
                        include_phase_candidates,
                    )
                    ranked_lists.append(cands[order[row, : len(cands)]])
            queries = [(q, None) for q in q_list]
            scores = _score(
                seq["poses"],
                queries,
                ranked_lists,
                [1, 5, 10],
                distance_threshold,
            )
            scores["n_queries"] = int(len(q_list))
            # Keep aliases compatible with evaluate_kitti_checkpoint outputs.
            scores["recall@1"] = float(scores["R@1"])
            scores["recall@5"] = float(scores["R@5"])
            scores["recall@10"] = float(scores["R@10"])
            out[seq["sequence"]] = scores
    if out:
        q_total = sum(v["n_queries"] for v in out.values())
        out["_average"] = {
            "recall@1": sum(v["recall@1"] * v["n_queries"] for v in out.values()) / q_total,
            "recall@5": sum(v["recall@5"] * v["n_queries"] for v in out.values()) / q_total,
            "recall@10": sum(v["recall@10"] * v["n_queries"] for v in out.values()) / q_total,
            "n_queries": q_total,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_kitti_phase_alignment_gat_fast.yaml")
    parser.add_argument("--encoder-preset", default="no_interdiff")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use-gated-context", action="store_true")
    parser.add_argument("--gate-initial-alpha", type=float, default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--train-sequences", nargs="+", default=["01", "02", "06", "07"])
    parser.add_argument("--val-sequences", nargs="+", default=["00", "05", "08"])
    parser.add_argument("--cache-dir", default="data/preprocessed_kitti_operating")
    parser.add_argument("--bev-cache-dir", default="data/preprocessed_kitti_bev_layout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layout-sectors", type=int, default=60)
    parser.add_argument("--bev-freqs", type=int, default=12)
    parser.add_argument("--bev-row-pool", type=int, default=16)
    parser.add_argument("--bev-row-pool-mode", default="max", choices=["max", "mean"])
    parser.add_argument("--bev-min-range", type=float, default=1.0)
    parser.add_argument("--bev-max-range", type=float, default=80.0)
    parser.add_argument("--bev-z-min", type=float, default=-3.0)
    parser.add_argument("--bev-z-max", type=float, default=5.0)
    parser.add_argument("--bev-height-layers", type=int, default=8)
    parser.add_argument("--bev-height-encoding", default="max", choices=["iris", "max", "physics3"])
    parser.add_argument("--temporal-edge-mode", default="bidirectional")
    parser.add_argument("--temporal-direction-mode", default="none")
    parser.add_argument("--similarity-min-k", type=int, default=0)
    parser.add_argument("--n-coarse", type=int, default=400)
    parser.add_argument("--max-candidates", type=int, default=800)
    parser.add_argument("--include-phase-candidates", action="store_true")
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-phase-weight", type=float, default=10.0)
    parser.add_argument("--base-embedding-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-residual-gate", action="store_true")
    parser.add_argument("--gate-hidden-dim", type=int, default=16)
    parser.add_argument("--residual-gate-initial-alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="results/kitti_learned_reranker.json")
    parser.add_argument("--checkpoint-out", default="results/kitti_learned_reranker.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = _apply_encoder_preset(_load_config(Path(args.config)), args.encoder_preset)
    # The reranker only needs the frozen embedding model. Disable phase-only
    # experimental heads so a simple 288D+128D checkpoint can be loaded from a
    # richer config file without requiring x_phase during embedding extraction.
    for key in (
        "phase_token",
        "phase_edge",
        "phase_alignment_edge",
        "phase_coherence",
        "dual_stream",
    ):
        config.get("gnn", {}).pop(key, None)
    if args.use_gated_context:
        config["gnn"]["use_residual_gate"] = True
        if args.gate_initial_alpha is not None:
            config["gnn"]["gate_initial_alpha"] = args.gate_initial_alpha
    root = Path(args.root or config["data"]["datasets"]["val"][0]["root"])
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    seq_kwargs = dict(
        root=root,
        config=config,
        checkpoint=checkpoint,
        cache_dir=Path(args.cache_dir),
        bev_cache_dir=Path(args.bev_cache_dir),
        device=args.device,
        layout_sectors=args.layout_sectors,
        bev_freqs=args.bev_freqs,
        bev_row_pool=args.bev_row_pool,
        bev_row_pool_mode=args.bev_row_pool_mode,
        temporal_edge_mode=args.temporal_edge_mode,
        temporal_direction_mode=args.temporal_direction_mode,
        similarity_min_k=args.similarity_min_k,
        bev_min_range=args.bev_min_range,
        bev_max_range=args.bev_max_range,
        bev_z_min=args.bev_z_min,
        bev_z_max=args.bev_z_max,
        bev_height_layers=args.bev_height_layers,
        bev_height_encoding=args.bev_height_encoding,
    )
    train_seqs = [_prepare_sequence(sequence=s, **seq_kwargs) for s in args.train_sequences]
    val_seqs = [_prepare_sequence(sequence=s, **seq_kwargs) for s in args.val_sequences]
    train_refs = _query_refs(train_seqs, args.distance_threshold, args.skip_frames)
    val_refs = _query_refs(val_seqs, args.distance_threshold, args.skip_frames)
    print(f"train_queries={len(train_refs)} val_queries={len(val_refs)}", flush=True)

    # Group refs by sequence so each batch can compute phase correlations with
    # one vectorized sketch tensor.
    train_by_seq: Dict[int, List[tuple[int, int]]] = {}
    for ref in train_refs:
        train_by_seq.setdefault(ref[0], []).append(ref)

    model = PhaseCorrelationReranker(
        n_shifts=args.layout_sectors,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        base_phase_weight=args.base_phase_weight,
        base_embedding_weight=args.base_embedding_weight,
        adaptive_residual_gate=args.adaptive_residual_gate,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_initial_alpha=args.residual_gate_initial_alpha,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1.0
    best_metrics = None
    metrics = _evaluate(
        model,
        val_seqs,
        val_refs,
        args.n_coarse,
        args.max_candidates,
        args.skip_frames,
        args.distance_threshold,
        args.include_phase_candidates,
        args.device,
        args.layout_sectors,
    )
    best = metrics["_average"]["recall@1"]
    best_metrics = metrics
    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "best_metrics": best_metrics,
        },
        args.checkpoint_out,
    )
    print(f"epoch=0 loss=nan val_r1={best:.4f} metrics={json.dumps(metrics)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        seq_ids = list(train_by_seq.keys())
        random.shuffle(seq_ids)
        for sid in seq_ids:
            refs = train_by_seq[sid][:]
            random.shuffle(refs)
            for start in range(0, len(refs), args.batch_size):
                batch_refs = refs[start : start + args.batch_size]
                if not batch_refs:
                    continue
                shift_corr, emb_sim, valid, labels = _make_batch(
                    train_seqs,
                    train_refs,
                    batch_refs,
                    args.n_coarse,
                    args.max_candidates,
                    args.skip_frames,
                    args.distance_threshold,
                    args.include_phase_candidates,
                    args.device,
                    args.layout_sectors,
                )
                logits = model(shift_corr, emb_sim, valid)
                loss = _listwise_loss(logits, labels, valid)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))

        metrics = _evaluate(
            model,
            val_seqs,
            val_refs,
            args.n_coarse,
            args.max_candidates,
            args.skip_frames,
            args.distance_threshold,
            args.include_phase_candidates,
            args.device,
            args.layout_sectors,
        )
        avg = metrics["_average"]["recall@1"]
        print(
            f"epoch={epoch} loss={np.mean(losses) if losses else float('nan'):.4f} "
            f"val_r1={avg:.4f} metrics={json.dumps(metrics)}",
            flush=True,
        )
        if avg > best:
            best = avg
            best_metrics = metrics
            Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_metrics": best_metrics,
                },
                args.checkpoint_out,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"best_recall@1": best, "best_metrics": best_metrics, "args": vars(args)}, f, indent=2)
    print(f"wrote {output} best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
