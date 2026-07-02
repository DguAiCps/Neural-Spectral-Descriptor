#!/usr/bin/env python3
"""Train cross-sensor learned phase-sketch reranker (KITTI+NCLT+HeLiPR+MulRan).

Mirrors `train_kitti_learned_reranker.py` but reads the train/val split from
the multi-dataset config (`data.datasets.{train,val}`). One sensor-agnostic
PhaseCorrelationReranker is shared across all four sensors.

Cache files for HeLiPR/MulRan are built on the fly via `_cross_sensor_cache`.
KITTI/NCLT use the same builder for a uniform layout under one cache dir.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _cross_sensor_cache import build_cache  # noqa: E402
from gnn.learned_reranker import PhaseCorrelationReranker  # noqa: E402
from run_kitti_bev_layout_rerank import _pool_rows  # noqa: E402
from run_kitti_operating_point import (  # noqa: E402
    _find_queries,
    _normalize,
    _score,
    _topk_cosine,
)
from evaluate_kitti_checkpoint import (  # noqa: E402
    _apply_encoder_preset,
    _load_config,
    _phase_sketch,
    _phase_sketch_keys,
)
from train_kitti_learned_reranker import (  # noqa: E402
    _candidate_set,
    _listwise_loss,
    _load_embeddings,
    _phase_corr_batch,
)


def _prepare_sequence(
    dataset_type: str,
    root: str,
    sequence: str,
    *,
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
    cylindrical_freqs: int = 0,
) -> Dict:
    op_path, bev_path = build_cache(
        dataset_type=dataset_type,
        root=root,
        sequence=sequence,
        config=config,
        cache_dir=cache_dir,
        bev_cache_dir=bev_cache_dir,
        device=device,
        layout_sectors=layout_sectors,
        scan_stride=1,
        bev_height_encoding=bev_height_encoding,
        bev_n_sectors=layout_sectors,
        bev_max_range=bev_max_range,
        bev_min_range=bev_min_range,
        bev_z_min=bev_z_min,
        bev_z_max=bev_z_max,
        bev_n_height_layers=bev_height_layers,
    )
    cache = np.load(op_path)
    bev_layouts = np.load(bev_path)["bev_layouts"]
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
        sensor_key=dataset_type,
    )
    sketch = _phase_sketch(bev_layouts.astype(np.float32), bev_freqs)
    if cylindrical_freqs > 0:
        # NSD operating cache holds the cylindrical/range-image layouts that the
        # encoder produced. Reuse them to build a complementary range phase
        # sketch and concatenate along the frequency axis so the existing
        # _phase_corr_batch / reranker pipeline stays unchanged.
        nsd_layouts = cache["nsd_layouts"].astype(np.float32)
        range_sketch = _phase_sketch(nsd_layouts, cylindrical_freqs)
        sketch = np.concatenate([sketch, range_sketch], axis=-1)
    return {
        "sensor": dataset_type,
        "sequence": sequence,
        "seq_key": f"{dataset_type}:{sequence}",
        "poses": cache["poses"].astype(np.float64),
        "embeddings": embeddings.astype(np.float32),
        "emb_norm": _normalize(embeddings),
        "phase_sketch": sketch,
        "phase_keys": _phase_sketch_keys(sketch),
    }


def _query_refs(seqs, distance_threshold, skip_frames):
    refs = []
    for sid, seq in enumerate(seqs):
        for q_idx, _ in _find_queries(seq["poses"], distance_threshold, skip_frames):
            refs.append((sid, int(q_idx)))
    return refs


def _make_batch(
    seqs,
    batch_refs,
    n_coarse,
    max_candidates,
    skip_frames,
    distance_threshold,
    include_phase_candidates,
    device,
    n_sectors,
):
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


@torch.no_grad()
def _hard_mine_select(scores, labels_pool, valid_pool, max_candidates):
    """Per-row stratified selection: keep all positives + top-score negatives.

    Returns (sel_idx_safe, valid_chosen) of shape (B, max_candidates).
    """
    B = scores.shape[0]
    device = scores.device
    sel_list = []
    valid_chosen = []
    for b in range(B):
        pos_mask = labels_pool[b] & valid_pool[b]
        neg_mask = (~labels_pool[b]) & valid_pool[b]
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
        n_pos = int(pos_idx.numel())
        if n_pos >= max_candidates:
            perm = torch.randperm(n_pos, device=device)[:max_candidates]
            chosen = pos_idx[perm]
            mask_b = torch.ones(max_candidates, dtype=torch.bool, device=device)
        else:
            n_neg_take = min(max_candidates - n_pos, int(neg_idx.numel()))
            if n_neg_take > 0:
                neg_scores = scores[b, neg_idx]
                order = torch.argsort(neg_scores, descending=True)
                top_neg = neg_idx[order[:n_neg_take]]
                chosen = torch.cat([pos_idx, top_neg])
            else:
                chosen = pos_idx
            pad_n = max_candidates - chosen.numel()
            if pad_n > 0:
                pad = torch.full((pad_n,), -1, dtype=torch.long, device=device)
                chosen = torch.cat([chosen, pad])
                mask_b = torch.cat([
                    torch.ones(max_candidates - pad_n, dtype=torch.bool, device=device),
                    torch.zeros(pad_n, dtype=torch.bool, device=device),
                ])
            else:
                mask_b = torch.ones(max_candidates, dtype=torch.bool, device=device)
        sel_list.append(chosen)
        valid_chosen.append(mask_b)
    sel = torch.stack(sel_list, dim=0)
    valid_chosen = torch.stack(valid_chosen, dim=0)
    return sel.clamp(min=0), valid_chosen


def _make_batch_mine(
    seqs,
    batch_refs,
    n_coarse,
    pool_size,
    max_candidates,
    skip_frames,
    distance_threshold,
    include_phase_candidates,
    device,
    n_sectors,
    mining_model=None,
):
    """Build a batch with optional hard-negative mining.

    If `mining_model` is given, builds a `pool_size` candidate pool, scores it
    with mining_model (no_grad, eval mode), and selects the top `max_candidates`
    using stratified mining (all positives + top-score negatives).
    """
    shift_corr_pool, emb_sim_pool, valid_pool, labels_pool = _make_batch(
        seqs,
        batch_refs,
        n_coarse=max(n_coarse, pool_size),
        max_candidates=pool_size,
        skip_frames=skip_frames,
        distance_threshold=distance_threshold,
        include_phase_candidates=include_phase_candidates,
        device=device,
        n_sectors=n_sectors,
    )
    if mining_model is None or pool_size <= max_candidates:
        return (
            shift_corr_pool[:, :max_candidates],
            emb_sim_pool[:, :max_candidates],
            valid_pool[:, :max_candidates],
            labels_pool[:, :max_candidates],
        )
    with torch.no_grad():
        was_training = mining_model.training
        mining_model.eval()
        scores = mining_model(shift_corr_pool, emb_sim_pool, valid_pool)
        if was_training:
            mining_model.train()
    scores = scores.masked_fill(~valid_pool, float("-inf"))
    sel, mined_valid = _hard_mine_select(scores, labels_pool, valid_pool, max_candidates)
    g_idx = sel.unsqueeze(-1).expand(-1, -1, n_sectors)
    shift_corr_out = torch.gather(shift_corr_pool, 1, g_idx)
    emb_sim_out = torch.gather(emb_sim_pool, 1, sel)
    valid_out = torch.gather(valid_pool, 1, sel) & mined_valid
    labels_out = torch.gather(labels_pool, 1, sel) & valid_out
    return (
        shift_corr_out * mined_valid.unsqueeze(-1),
        emb_sim_out * mined_valid,
        valid_out,
        labels_out,
    )


def _evaluate(
    model,
    seqs,
    refs,
    n_coarse,
    max_candidates,
    skip_frames,
    distance_threshold,
    include_phase_candidates,
    device,
    n_sectors,
):
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
            scores = _score(seq["poses"], queries, ranked_lists, [1, 5, 10], distance_threshold)
            scores["n_queries"] = int(len(q_list))
            scores["recall@1"] = float(scores["R@1"])
            scores["recall@5"] = float(scores["R@5"])
            scores["recall@10"] = float(scores["R@10"])
            out[seq["seq_key"]] = scores
    if out:
        q_total = sum(v["n_queries"] for v in out.values())
        macro_r1 = float(np.mean([v["recall@1"] for v in out.values()]))
        macro_r5 = float(np.mean([v["recall@5"] for v in out.values()]))
        macro_r10 = float(np.mean([v["recall@10"] for v in out.values()]))
        out["_average"] = {
            "recall@1": sum(v["recall@1"] * v["n_queries"] for v in out.values()) / q_total,
            "recall@5": sum(v["recall@5"] * v["n_queries"] for v in out.values()) / q_total,
            "recall@10": sum(v["recall@10"] * v["n_queries"] for v in out.values()) / q_total,
            "n_queries": q_total,
        }
        out["_macro"] = {
            "recall@1": macro_r1,
            "recall@5": macro_r5,
            "recall@10": macro_r10,
            "n_sequences": len([k for k in out.keys() if not k.startswith("_")]),
        }
    return out


def _collect_split_specs(
    config: Dict,
    split: str,
    exclude: List[str] | None = None,
) -> List[Tuple[str, str, str]]:
    exclude_set = set(exclude or [])
    specs = []
    for entry in config["data"]["datasets"].get(split, []) or []:
        ds_type = entry["type"]
        root = entry["root"]
        for seq in entry["sequences"]:
            key = f"{ds_type}:{seq}"
            if key in exclude_set:
                print(f"skip {key} (excluded)", flush=True)
                continue
            specs.append((ds_type, root, seq))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_multi_dataset.yaml")
    parser.add_argument("--encoder-preset", default="no_interdiff")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use-gated-context", action="store_true")
    parser.add_argument("--gate-initial-alpha", type=float, default=None)
    parser.add_argument("--cache-dir", default="data/preprocessed_cross_sensor_operating")
    parser.add_argument("--bev-cache-dir", default="data/preprocessed_cross_sensor_bev_layout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layout-sectors", type=int, default=60)
    parser.add_argument("--bev-freqs", type=int, default=12)
    parser.add_argument(
        "--cylindrical-freqs",
        type=int,
        default=0,
        help="Number of low frequencies of cylindrical/range phase to concat with BEV sketch (0 = BEV-only).",
    )
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--enable-hard-mining", action="store_true")
    parser.add_argument("--mining-pool-size", type=int, default=1600)
    parser.add_argument("--mine-from-epoch", type=int, default=5)
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
    parser.add_argument("--output", default="results/cross_sensor_learned_reranker.json")
    parser.add_argument("--checkpoint-out", default="results/cross_sensor_learned_reranker.pth")
    parser.add_argument(
        "--exclude-sequences",
        nargs="*",
        default=[],
        help="dataset:sequence pairs to skip (memory-heavy sequences), e.g. mulran:Sejong01",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = _apply_encoder_preset(_load_config(Path(args.config)), args.encoder_preset)
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

    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    seq_kwargs = dict(
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
        cylindrical_freqs=args.cylindrical_freqs,
    )

    train_specs = _collect_split_specs(config, "train", exclude=args.exclude_sequences)
    val_specs = _collect_split_specs(config, "val", exclude=args.exclude_sequences)
    print(f"train specs: {len(train_specs)} | val specs: {len(val_specs)}", flush=True)

    train_seqs = []
    for ds_type, root, seq in train_specs:
        print(f"prep train {ds_type}:{seq}", flush=True)
        train_seqs.append(_prepare_sequence(ds_type, root, seq, **seq_kwargs))

    val_seqs = []
    for ds_type, root, seq in val_specs:
        print(f"prep val {ds_type}:{seq}", flush=True)
        val_seqs.append(_prepare_sequence(ds_type, root, seq, **seq_kwargs))

    train_refs = _query_refs(train_seqs, args.distance_threshold, args.skip_frames)
    val_refs = _query_refs(val_seqs, args.distance_threshold, args.skip_frames)
    print(f"train_queries={len(train_refs)} val_queries={len(val_refs)}", flush=True)

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
        {"model_state_dict": model.state_dict(), "args": vars(args), "best_metrics": best_metrics},
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
                mining_model_now = (
                    model
                    if args.enable_hard_mining and epoch >= args.mine_from_epoch
                    else None
                )
                shift_corr, emb_sim, valid, labels = _make_batch_mine(
                    train_seqs,
                    batch_refs,
                    args.n_coarse,
                    args.mining_pool_size,
                    args.max_candidates,
                    args.skip_frames,
                    args.distance_threshold,
                    args.include_phase_candidates,
                    args.device,
                    args.layout_sectors,
                    mining_model=mining_model_now,
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
            torch.save(
                {"model_state_dict": model.state_dict(), "args": vars(args), "best_metrics": best_metrics},
                args.checkpoint_out,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"best_recall@1": best, "best_metrics": best_metrics, "args": vars(args)}, f, indent=2)
    print(f"wrote {output} best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
