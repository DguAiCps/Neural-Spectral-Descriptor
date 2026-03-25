"""
Quantization Module for Spectral Histograms

Compresses 16-bin float32 histograms to 16-bit integers with metadata:
- 32 bytes: quantized histogram (16 bins × 16-bit)
- 120 bytes: metadata (pose, timestamp, hash)
- Total: 152 bytes per keyframe

Key requirements:
- Preserve histogram sum normalization
- Reversible quantization for retrieval
- Compact binary serialization

TODO: Update CompressedDescriptor and HistogramQuantizer for multi-stat descriptors.
Currently hardcoded for 16-bin single-stat (256D → 32 bytes uint16).
Multi-stat (e.g., mean+std = 512D) would need 1024 bytes uint16.
Training uses float32 .npz cache directly, so this only affects inference deployment.
"""

import numpy as np
import struct
import hashlib
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CompressedDescriptor:
    """
    Compressed keyframe descriptor (152 bytes total)

    Format:
        - histogram: 16 × uint16 (32 bytes)
        - pose: 7 × float32 (28 bytes) [x, y, z, qw, qx, qy, qz]
        - timestamp: float64 (8 bytes)
        - keyframe_id: uint32 (4 bytes)
        - hash: 20 bytes (SHA-1 of point cloud)
        - reserved: 60 bytes (for future use)
    """
    histogram: np.ndarray  # (16,) uint16
    pose: np.ndarray  # (7,) [x, y, z, qw, qx, qy, qz]
    timestamp: float
    keyframe_id: int
    point_cloud_hash: bytes  # 20 bytes SHA-1

    def to_bytes(self) -> bytes:
        """
        Serialize descriptor to 152 bytes

        Returns:
            152-byte binary representation
        """
        # Histogram: 16 × 2 bytes = 32 bytes
        hist_bytes = self.histogram.astype(np.uint16).tobytes()

        # Pose: 7 × 4 bytes = 28 bytes
        pose_bytes = self.pose.astype(np.float32).tobytes()

        # Timestamp: 8 bytes
        timestamp_bytes = struct.pack('d', self.timestamp)

        # Keyframe ID: 4 bytes
        id_bytes = struct.pack('I', self.keyframe_id)

        # Hash: 20 bytes
        hash_bytes = self.point_cloud_hash

        # Reserved: 60 bytes (zeros)
        reserved = bytes(60)

        # Concatenate
        total = hist_bytes + pose_bytes + timestamp_bytes + id_bytes + hash_bytes + reserved

        assert len(total) == 152, f"Expected 152 bytes, got {len(total)}"

        return total

    @staticmethod
    def from_bytes(data: bytes) -> 'CompressedDescriptor':
        """
        Deserialize descriptor from 152 bytes

        Args:
            data: 152-byte binary representation

        Returns:
            CompressedDescriptor instance
        """
        assert len(data) == 152, f"Expected 152 bytes, got {len(data)}"

        # Parse histogram: 32 bytes
        histogram = np.frombuffer(data[:32], dtype=np.uint16)

        # Parse pose: 28 bytes
        pose = np.frombuffer(data[32:60], dtype=np.float32)

        # Parse timestamp: 8 bytes
        timestamp = struct.unpack('d', data[60:68])[0]

        # Parse keyframe ID: 4 bytes
        keyframe_id = struct.unpack('I', data[68:72])[0]

        # Parse hash: 20 bytes
        point_cloud_hash = data[72:92]

        # Reserved: 60 bytes (ignored)

        return CompressedDescriptor(
            histogram=histogram,
            pose=pose,
            timestamp=timestamp,
            keyframe_id=keyframe_id,
            point_cloud_hash=point_cloud_hash
        )


class HistogramQuantizer:
    """
    Quantizes normalized histograms to 16-bit unsigned integers
    """

    def __init__(self, n_bins: int = 16, epsilon: float = 1e-8):
        """
        Initialize quantizer

        Args:
            n_bins: Number of histogram bins (16)
            epsilon: Small constant for numerical stability
        """
        self.n_bins = n_bins
        self.epsilon = epsilon

        # Maximum value for uint16
        self.max_value = 65535

    def quantize(self, histogram: np.ndarray) -> np.ndarray:
        """
        Quantize normalized histogram to uint16

        Args:
            histogram: (n_bins,) normalized histogram (sums to 1.0)

        Returns:
            (n_bins,) quantized histogram as uint16
        """
        assert len(histogram) == self.n_bins, f"Expected {self.n_bins} bins, got {len(histogram)}"

        # Ensure normalized
        hist_sum = histogram.sum()
        if hist_sum > self.epsilon:
            histogram = histogram / (hist_sum + self.epsilon)

        # Scale to uint16 range
        # Reserve max_value for normalization
        quantized = np.round(histogram * self.max_value).astype(np.uint16)

        # Ensure sum is preserved by renormalization
        # This prevents rounding errors from accumulating
        quantized_sum = quantized.sum()
        if quantized_sum > 0:
            # Distribute rounding error
            error = self.max_value - quantized_sum

            if error != 0:
                # Add error to largest bin (least relative impact)
                max_idx = quantized.argmax()
                quantized[max_idx] = np.clip(
                    quantized[max_idx] + error,
                    0,
                    self.max_value
                ).astype(np.uint16)

        return quantized

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """
        Dequantize uint16 histogram to normalized float32

        Args:
            quantized: (n_bins,) uint16 histogram

        Returns:
            (n_bins,) normalized float32 histogram
        """
        assert len(quantized) == self.n_bins, f"Expected {self.n_bins} bins, got {len(quantized)}"

        # Convert to float and normalize
        histogram = quantized.astype(np.float32)

        hist_sum = histogram.sum()
        if hist_sum > self.epsilon:
            histogram = histogram / (hist_sum + self.epsilon)
        else:
            # Uniform distribution if empty
            histogram = np.ones(self.n_bins, dtype=np.float32) / self.n_bins

        return histogram


def compute_point_cloud_hash(points: np.ndarray) -> bytes:
    """
    Compute SHA-1 hash of point cloud for identification

    Args:
        points: (N, 3) or (N, 4) point cloud

    Returns:
        20-byte SHA-1 hash
    """
    # Use only XYZ coordinates for hash (ignore intensity)
    xyz = points[:, :3].astype(np.float32)

    # Compute SHA-1
    hasher = hashlib.sha1()
    hasher.update(xyz.tobytes())

    return hasher.digest()


def pose_to_7dof(pose: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) matrix to 7-DOF representation [x, y, z, qw, qx, qy, qz]

    Args:
        pose: (4, 4) SE(3) transformation matrix

    Returns:
        (7,) array [x, y, z, qw, qx, qy, qz]
    """
    from scipy.spatial.transform import Rotation

    # Extract translation
    translation = pose[:3, 3]

    # Extract rotation and convert to quaternion
    rotation_matrix = pose[:3, :3]
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # [x, y, z, w] in scipy

    # Reorder to [w, x, y, z]
    quaternion_wxyz = np.array([
        quaternion[3],  # w
        quaternion[0],  # x
        quaternion[1],  # y
        quaternion[2]   # z
    ])

    # Concatenate
    pose_7dof = np.concatenate([translation, quaternion_wxyz])

    return pose_7dof


def pose_from_7dof(pose_7dof: np.ndarray) -> np.ndarray:
    """
    Convert 7-DOF representation to SE(3) matrix

    Args:
        pose_7dof: (7,) array [x, y, z, qw, qx, qy, qz]

    Returns:
        (4, 4) SE(3) transformation matrix
    """
    from scipy.spatial.transform import Rotation

    # Extract translation
    translation = pose_7dof[:3]

    # Extract quaternion [w, x, y, z]
    quaternion_wxyz = pose_7dof[3:]

    # Reorder to [x, y, z, w] for scipy
    quaternion_xyzw = np.array([
        quaternion_wxyz[1],  # x
        quaternion_wxyz[2],  # y
        quaternion_wxyz[3],  # z
        quaternion_wxyz[0]   # w
    ])

    # Convert to rotation matrix
    rotation = Rotation.from_quat(quaternion_xyzw)
    rotation_matrix = rotation.as_matrix()

    # Construct SE(3) matrix
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = translation

    return pose


def compress_descriptor(
    histogram: np.ndarray,
    pose: np.ndarray,
    timestamp: float,
    keyframe_id: int,
    points: np.ndarray
) -> CompressedDescriptor:
    """
    Compress keyframe descriptor to 152 bytes

    Args:
        histogram: (16,) normalized histogram
        pose: (4, 4) SE(3) pose
        timestamp: Timestamp in seconds
        keyframe_id: Unique keyframe ID
        points: (N, 3) or (N, 4) point cloud

    Returns:
        CompressedDescriptor (152 bytes)
    """
    quantizer = HistogramQuantizer()

    # Quantize histogram
    quantized_hist = quantizer.quantize(histogram)

    # Convert pose to 7-DOF
    pose_7dof = pose_to_7dof(pose)

    # Compute point cloud hash
    pc_hash = compute_point_cloud_hash(points)

    # Create descriptor
    descriptor = CompressedDescriptor(
        histogram=quantized_hist,
        pose=pose_7dof,
        timestamp=timestamp,
        keyframe_id=keyframe_id,
        point_cloud_hash=pc_hash
    )

    return descriptor


def decompress_descriptor(
    descriptor: CompressedDescriptor
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Decompress descriptor to usable format

    Args:
        descriptor: CompressedDescriptor

    Returns:
        histogram: (16,) normalized float32 histogram
        pose: (4, 4) SE(3) pose
        timestamp: float
        keyframe_id: int
    """
    quantizer = HistogramQuantizer()

    # Dequantize histogram
    histogram = quantizer.dequantize(descriptor.histogram)

    # Convert pose from 7-DOF
    pose = pose_from_7dof(descriptor.pose)

    return histogram, pose, descriptor.timestamp, descriptor.keyframe_id


def test_quantization_error(histogram: np.ndarray, n_trials: int = 100) -> Dict[str, float]:
    """
    Test quantization error

    Args:
        histogram: (16,) normalized histogram
        n_trials: Number of test iterations

    Returns:
        Dictionary with error statistics
    """
    quantizer = HistogramQuantizer()

    errors = []

    for _ in range(n_trials):
        # Quantize and dequantize
        quantized = quantizer.quantize(histogram)
        dequantized = quantizer.dequantize(quantized)

        # Compute error
        error = np.abs(histogram - dequantized).max()
        errors.append(error)

    return {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'std_error': np.std(errors)
    }
