"""Point cloud registration algorithms and pose graph optimization."""

import numpy as np
import open3d as o3d
from typing import Optional, List, Tuple
from core.transforms import is_valid_transformation
from core.pointcloud import preprocess_point_cloud


def colored_icp_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    max_correspondence_distance: float,
    init_transform: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Tuple[
    Optional[np.ndarray],
    Optional[o3d.pipelines.registration.RegistrationResult]
]:
    """
    Perform colored ICP registration between source and target point clouds.
    
    Args:
        source: Source Open3D PointCloud
        target: Target Open3D PointCloud
        voxel_size: Voxel size for processing
        max_correspondence_distance: Maximum distance for correspondences
        init_transform: Initial 4x4 transformation (optional)
        
    Returns:
        Tuple of (transformation, result) or (None, None) if failed
    """
    try:
        if init_transform is None:
            init_transform = np.identity(4)
        
        result = o3d.pipelines.registration.registration_colored_icp(
            source, target, max_correspondence_distance, init_transform,
            o3d.pipelines.registration.
            TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=50))
        return result.transformation, result
    except Exception as e:
        if verbose:
            print(f"    ⚠ Colored ICP failed: {e}")
        return None, None


def full_registration(
    pcds_down_standard: List[o3d.geometry.PointCloud],
    voxel_size: float,
    max_correspondence_distance: float,
    initial_poses: Optional[List[np.ndarray]] = None,
    min_fitness_consecutive: float = 0.75,
    min_fitness_loop: float = 0.4,
    loop_closure_interval: int = 15,
    loop_closure_distance_threshold: int = 5,
    verbose: bool = False
) -> o3d.pipelines.registration.PoseGraph:
    """
    Perform full pairwise registration with odometry edges and loop closures.
    
    Args:
        pcds_down_standard: List of preprocessed point clouds
        voxel_size: Voxel size for registration
        max_correspondence_distance: Maximum correspondence distance
        initial_poses: Optional list of initial poses for odometry
            initialization
        min_fitness_consecutive: Minimum fitness for consecutive
            frame registration
        min_fitness_loop: Minimum fitness for loop closure
        loop_closure_interval: Interval for attempting loop closures
        loop_closure_distance_threshold: Minimum frame distance for
            loop closure
        verbose: Whether to print verbose output
        
    Returns:
        PoseGraph object
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(odometry))
    
    n_pcds = len(pcds_down_standard)
    print(f"\nPairwise registration for {n_pcds} point clouds "
          f"(fragment/local)...")
    
    consecutive_success = 0
    consecutive_failed = 0
    loop_attempted = 0
    loop_success = 0
    
    # Consecutive frame registration
    for source_id in range(n_pcds - 1):
        target_id = source_id + 1
        if verbose:
            print(f"  Odometry {source_id} → {target_id} "
                  f"(colored ICP)", end=" ")
        
        # Compute initial transform from odometry if available
        init_transform = None
        if initial_poses is not None and len(initial_poses) > target_id:
            T_source = initial_poses[source_id]
            T_target = initial_poses[target_id]
            init_transform = np.linalg.inv(T_source) @ T_target
            if verbose:
                print("[init from odom] ", end="")
        
        transformation, result = colored_icp_registration(
            pcds_down_standard[source_id], pcds_down_standard[target_id],
            voxel_size, max_correspondence_distance, init_transform)
        
        if transformation is not None and result is not None:
            fitness = result.fitness
            if (fitness >= min_fitness_consecutive and
                    is_valid_transformation(transformation)):
                if verbose:
                    print(f"✓ fitness={fitness:.3f}, "
                          f"rmse={result.inlier_rmse:.3f}")
                consecutive_success += 1
                odometry = np.dot(transformation, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id, target_id, transformation,
                        information=np.identity(6) * fitness,
                        uncertain=False))
            else:
                if verbose:
                    print(f"✗ fitness={fitness:.3f} (low/invalid)")
                consecutive_failed += 1
                transformation = np.identity(4)
                odometry = np.dot(transformation, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id, target_id, transformation,
                        information=np.identity(6) * 0.01,
                        uncertain=True))
        else:
            if verbose:
                print("✗ registration failed")
            consecutive_failed += 1
            transformation = np.identity(4)
            odometry = np.dot(transformation, odometry)
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.linalg.inv(odometry)))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_id, target_id, transformation,
                    information=np.identity(6) * 0.01,
                    uncertain=True))
    
    # Loop closure detection
    print(f"\n  Adding sparse loop closures...")
    for source_id in range(0, n_pcds, loop_closure_interval):
        for target_id in range(
                source_id + loop_closure_distance_threshold,
                n_pcds, loop_closure_interval):
            if target_id >= n_pcds:
                break
            loop_attempted += 1
            frame_distance = target_id - source_id
            if verbose:
                print(f"  Loop {source_id} → {target_id} "
                      f"(distance={frame_distance})", end=" ")
            
            transformation, result = colored_icp_registration(
                pcds_down_standard[source_id], pcds_down_standard[target_id],
                voxel_size, max_correspondence_distance)
            
            if transformation is not None and result is not None:
                fitness = result.fitness
                if (fitness >= min_fitness_loop and
                        is_valid_transformation(
                            transformation, max_translation=500.0)):
                    if verbose:
                        print(f"✓ fitness={fitness:.3f}, "
                              f"rmse={result.inlier_rmse:.3f}")
                    loop_success += 1
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            source_id, target_id, transformation,
                            information=np.identity(6) * fitness,
                            uncertain=True))
                else:
                    if verbose:
                        print(f"✗ fitness={fitness:.3f} (low/invalid)")
            else:
                if verbose:
                    print("✗ failed")

    print(f"\n✓ Local registration complete: "
          f"{consecutive_success} consecutive OK, "
          f"{loop_success} / {loop_attempted} loop closures OK")
    return pose_graph


def optimize_pose_graph(
    pose_graph: o3d.pipelines.registration.PoseGraph,
    max_correspondence_distance: float,
    preference_loop_closure: float = 1.0
) -> o3d.pipelines.registration.PoseGraph:
    """
    Optimize pose graph using global optimization.
    
    Args:
        pose_graph: PoseGraph to optimize
        max_correspondence_distance: Maximum correspondence distance
        preference_loop_closure: Weight for loop closure edges
        
    Returns:
        Optimized PoseGraph
    """
    print("\nOptimizing pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.35,
        preference_loop_closure=preference_loop_closure,
        reference_node=0)
    
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    
    print("Pose graph optimization complete!")
    return pose_graph
