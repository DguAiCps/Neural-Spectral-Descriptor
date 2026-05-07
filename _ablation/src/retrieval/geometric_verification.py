"""
Geometric Verification using ICP/GICP

Verifies loop closure candidates using geometric alignment:
- GICP (Generalized ICP) for robust registration
- Fitness threshold: >0.3
- RMSE threshold: <0.5m
- Returns relative pose and information matrix
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict


class GeometricVerifier:
    """
    Verifies loop closures using ICP/GICP registration
    """

    def __init__(
        self,
        method: str = "gicp",
        fitness_threshold: float = 0.3,
        rmse_threshold: float = 0.5,
        max_iterations: int = 30,
        voxel_downsample: float = 0.3,
        max_correspondence_distance: float = 1.0
    ):
        """
        Initialize geometric verifier

        Args:
            method: Registration method ("icp" or "gicp")
            fitness_threshold: Minimum fitness score (fraction of inliers)
            rmse_threshold: Maximum RMSE in meters
            max_iterations: Maximum ICP iterations
            voxel_downsample: Voxel size for downsampling (meters)
            max_correspondence_distance: Max distance for correspondences
        """
        self.method = method
        self.fitness_threshold = fitness_threshold
        self.rmse_threshold = rmse_threshold
        self.max_iterations = max_iterations
        self.voxel_downsample = voxel_downsample
        self.max_correspondence_distance = max_correspondence_distance

    def verify(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray], Dict]:
        """
        Verify and refine loop closure using ICP/GICP

        Args:
            source_points: (N, 3) source point cloud
            target_points: (M, 3) target point cloud
            initial_transform: (4, 4) initial transformation guess (identity if None)

        Returns:
            verified: True if verification passed
            transform: (4, 4) refined transformation (None if failed)
            info: Dictionary with fitness, RMSE, and information matrix
        """
        # Convert to Open3D point clouds
        source_pcd = self._numpy_to_o3d(source_points)
        target_pcd = self._numpy_to_o3d(target_points)

        # Downsample
        if self.voxel_downsample > 0:
            source_pcd = source_pcd.voxel_down_sample(self.voxel_downsample)
            target_pcd = target_pcd.voxel_down_sample(self.voxel_downsample)

        # Initial transformation
        if initial_transform is None:
            initial_transform = np.eye(4)

        # Estimate normals (required for GICP)
        if self.method == "gicp":
            source_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_downsample * 2,
                    max_nn=30
                )
            )
            target_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_downsample * 2,
                    max_nn=30
                )
            )

        # Run ICP/GICP
        if self.method == "gicp":
            result = o3d.pipelines.registration.registration_generalized_icp(
                source_pcd,
                target_pcd,
                self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations
                )
            )
        else:  # icp
            result = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations
                )
            )

        # Extract results
        transform = np.array(result.transformation)
        fitness = result.fitness
        rmse = result.inlier_rmse

        # Compute information matrix (for pose graph optimization)
        information_matrix = self._compute_information_matrix(
            source_pcd,
            target_pcd,
            transform,
            fitness
        )

        # Check thresholds
        verified = (fitness >= self.fitness_threshold) and (rmse <= self.rmse_threshold)

        info = {
            'fitness': fitness,
            'rmse': rmse,
            'information_matrix': information_matrix,
            'num_iterations': result.fitness  # Not directly available in result
        }

        if verified:
            return True, transform, info
        else:
            return False, None, info

    def _numpy_to_o3d(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Convert numpy array to Open3D point cloud

        Args:
            points: (N, 3) or (N, 4) numpy array

        Returns:
            Open3D PointCloud
        """
        pcd = o3d.geometry.PointCloud()

        # Use only XYZ coordinates
        if points.shape[1] >= 3:
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        return pcd

    def _compute_information_matrix(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        transform: np.ndarray,
        fitness: float
    ) -> np.ndarray:
        """
        Compute information matrix for pose graph optimization

        Simple heuristic based on fitness and point cloud size.
        Higher fitness -> higher information (more confident)

        Args:
            source_pcd: Source point cloud
            target_pcd: Target point cloud
            transform: Transformation matrix
            fitness: ICP fitness score

        Returns:
            (6, 6) information matrix
        """
        # Simple heuristic: diagonal information matrix
        # Scale by fitness (higher fitness = more confident)

        # Base information values
        translation_info = 100.0  # Information for translation
        rotation_info = 1000.0  # Information for rotation

        # Scale by fitness
        scale = fitness

        # Create diagonal information matrix
        # Order: [x, y, z, roll, pitch, yaw]
        information = np.eye(6)
        information[:3, :3] *= translation_info * scale  # Translation
        information[3:, 3:] *= rotation_info * scale  # Rotation

        return information


def verify_loop_closure(
    source_points: np.ndarray,
    target_points: np.ndarray,
    method: str = "gicp",
    fitness_threshold: float = 0.3,
    rmse_threshold: float = 0.5
) -> Tuple[bool, Optional[np.ndarray], Dict]:
    """
    Convenience function to verify loop closure

    Args:
        source_points: (N, 3) source point cloud
        target_points: (M, 3) target point cloud
        method: Registration method ("icp" or "gicp")
        fitness_threshold: Minimum fitness score
        rmse_threshold: Maximum RMSE

    Returns:
        verified: True if verification passed
        transform: (4, 4) transformation (None if failed)
        info: Dictionary with metrics
    """
    verifier = GeometricVerifier(
        method=method,
        fitness_threshold=fitness_threshold,
        rmse_threshold=rmse_threshold
    )

    return verifier.verify(source_points, target_points)


def batch_verify_candidates(
    query_points: np.ndarray,
    candidate_points_list: list,
    method: str = "gicp",
    fitness_threshold: float = 0.3,
    rmse_threshold: float = 0.5,
    parallel: bool = False
) -> list:
    """
    Verify multiple loop closure candidates

    Args:
        query_points: (N, 3) query point cloud
        candidate_points_list: List of (M, 3) candidate point clouds
        method: Registration method
        fitness_threshold: Minimum fitness
        rmse_threshold: Maximum RMSE
        parallel: Use parallel verification (future enhancement)

    Returns:
        List of (verified, transform, info) tuples
    """
    verifier = GeometricVerifier(
        method=method,
        fitness_threshold=fitness_threshold,
        rmse_threshold=rmse_threshold
    )

    results = []

    for candidate_points in candidate_points_list:
        result = verifier.verify(query_points, candidate_points)
        results.append(result)

    return results


def compute_pose_graph_edge(
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    relative_transform: np.ndarray,
    information_matrix: np.ndarray
) -> Dict:
    """
    Compute pose graph edge for g2o format

    Args:
        source_pose: (4, 4) source pose in world frame
        target_pose: (4, 4) target pose in world frame
        relative_transform: (4, 4) measured relative transformation
        information_matrix: (6, 6) information matrix

    Returns:
        Dictionary with edge information for g2o
    """
    from data.pose_utils import pose_to_7dof

    # Convert poses to 7-DOF representation
    source_7dof = pose_to_7dof(source_pose)
    target_7dof = pose_to_7dof(target_pose)
    relative_7dof = pose_to_7dof(relative_transform)

    edge = {
        'source_id': 0,  # To be filled by caller
        'target_id': 0,  # To be filled by caller
        'relative_pose': relative_7dof,
        'information_matrix': information_matrix
    }

    return edge


def save_loop_closures_g2o(
    loop_closures: list,
    output_path: str
):
    """
    Save loop closures in g2o format

    Args:
        loop_closures: List of loop closure dictionaries
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for lc in loop_closures:
            source_id = lc['source_id']
            target_id = lc['target_id']
            pose = lc['relative_pose']  # [x, y, z, qw, qx, qy, qz]
            info = lc['information_matrix']  # (6, 6)

            # g2o EDGE_SE3:QUAT format
            # EDGE_SE3:QUAT id1 id2 x y z qx qy qz qw info(0,0) info(0,1) ... info(5,5)

            # Convert quaternion from [w,x,y,z] to [x,y,z,w] for g2o
            x, y, z = pose[0], pose[1], pose[2]
            qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]

            # Write edge
            f.write(f"EDGE_SE3:QUAT {source_id} {target_id} ")
            f.write(f"{x} {y} {z} {qx} {qy} {qz} {qw} ")

            # Write upper triangular part of information matrix
            for i in range(6):
                for j in range(i, 6):
                    f.write(f"{info[i, j]} ")

            f.write("\n")

    print(f"Saved {len(loop_closures)} loop closures to {output_path}")
