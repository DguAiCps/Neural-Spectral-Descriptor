"""Tests for triplet mining (hard negative selection)."""

import numpy as np
import pytest


class TestTripletMinerBasic:
    """Test TripletMiner with synthetic trajectory data."""

    @pytest.fixture
    def miner(self):
        from gnn.triplet_miner import TripletMiner
        return TripletMiner(
            positive_distance_max=10.0,
            negative_distance_min=20.0,
            positive_temporal_min=3,
            negative_temporal_min=3,
            mining_strategy='hard',
        )

    @pytest.fixture
    def straight_line_data(self):
        """50 keyframes, 2m apart, straight line."""
        n = 50
        descriptors = np.random.randn(n, 64).astype(np.float32)
        # Normalize for cosine
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)

        poses = np.zeros((n, 4, 4))
        for i in range(n):
            poses[i] = np.eye(4)
            poses[i, 0, 3] = i * 2.0  # 2m apart

        sequence_ids = np.zeros(n, dtype=np.int64)
        return descriptors, poses, sequence_ids

    def test_mine_returns_triplets(self, miner, straight_line_data):
        descriptors, poses, seq_ids = straight_line_data
        triplets = miner.mine_triplets(
            descriptors, poses, n_triplets_per_anchor=1,
            sequence_ids=seq_ids, parallel=False,
        )
        assert len(triplets) > 0

    def test_triplet_positive_distance(self, miner, straight_line_data):
        """Positive should be within positive_distance_max."""
        descriptors, poses, seq_ids = straight_line_data
        triplets = miner.mine_triplets(
            descriptors, poses, n_triplets_per_anchor=1,
            sequence_ids=seq_ids, parallel=False,
        )
        for anchor, pos, neg in triplets:
            pos_dist = np.linalg.norm(poses[anchor][:3, 3] - poses[pos][:3, 3])
            assert pos_dist <= miner.positive_distance_max + 1e-6

    def test_triplet_negative_distance(self, miner, straight_line_data):
        """Negative should be beyond negative_distance_min."""
        descriptors, poses, seq_ids = straight_line_data
        triplets = miner.mine_triplets(
            descriptors, poses, n_triplets_per_anchor=1,
            sequence_ids=seq_ids, parallel=False,
        )
        for anchor, pos, neg in triplets:
            neg_dist = np.linalg.norm(poses[anchor][:3, 3] - poses[neg][:3, 3])
            assert neg_dist >= miner.negative_distance_min - 1e-6

    def test_triplet_temporal_gap(self, miner, straight_line_data):
        """Positive and negative should have temporal gap."""
        descriptors, poses, seq_ids = straight_line_data
        triplets = miner.mine_triplets(
            descriptors, poses, n_triplets_per_anchor=1,
            sequence_ids=seq_ids, parallel=False,
        )
        for anchor, pos, neg in triplets:
            assert abs(anchor - pos) >= miner.positive_temporal_min
            assert abs(anchor - neg) >= miner.negative_temporal_min

    def test_no_cross_sequence_triplets(self):
        """Triplets should stay within the same sequence."""
        from gnn.triplet_miner import TripletMiner
        miner = TripletMiner(
            positive_distance_max=5.0,
            negative_distance_min=10.0,
            positive_temporal_min=3,
            negative_temporal_min=3,
        )
        n = 30
        # Two sequences, widely separated
        descriptors = np.random.randn(n, 64).astype(np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
        poses = np.zeros((n, 4, 4))
        seq_ids = np.zeros(n, dtype=np.int64)
        for i in range(n):
            poses[i] = np.eye(4)
            if i < 15:
                poses[i, 0, 3] = i * 2.0
                seq_ids[i] = 0
            else:
                poses[i, 0, 3] = i * 2.0 + 1000  # far away
                seq_ids[i] = 1

        triplets = miner.mine_triplets(
            descriptors, poses, n_triplets_per_anchor=1,
            sequence_ids=seq_ids, parallel=False,
        )
        for anchor, pos, neg in triplets:
            assert seq_ids[anchor] == seq_ids[pos]
            assert seq_ids[anchor] == seq_ids[neg]

    def test_too_few_points(self):
        """With < 3 points in sequence, should return empty."""
        from gnn.triplet_miner import TripletMiner
        miner = TripletMiner()
        desc = np.random.randn(2, 64).astype(np.float32)
        poses = np.stack([np.eye(4), np.eye(4)])
        seq_ids = np.array([0, 0])
        triplets = miner.mine_triplets(
            desc, poses, n_triplets_per_anchor=1, sequence_ids=seq_ids,
        )
        assert len(triplets) == 0


class TestBatchTripletMiner:

    def test_import(self):
        from gnn.triplet_miner import BatchTripletMiner
        miner = BatchTripletMiner()
        assert miner is not None
