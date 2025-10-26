"""Fragment-based processing for large-scale 3D reconstruction."""

import numpy as np
import open3d as o3d
import copy
from typing import Optional, List, Tuple
from registration import (full_registration, optimize_pose_graph, 
                          colored_icp_registration, preprocess_point_cloud)
from transforms import is_valid_transformation
from config import VOXEL_SIZE, MAX_CORRESPONDENCE_DISTANCE


def split_into_fragments(
    n: int,
    frag_size: int,
    overlap: int,
    timestamps: Optional[List[int]] = None,
    gap_threshold_ns: float = 1e9
) -> List[Tuple[int, int]]:
    """
    Split data into overlapping fragments, optionally detecting time gaps.
    
    Args:
        n: Total number of items
        frag_size: Maximum size of each fragment
        overlap: Overlap between consecutive fragments
        timestamps: Optional list of timestamps (in nanoseconds)
        gap_threshold_ns: Threshold for detecting time gaps (default 1s)
        
    Returns:
        List of (start, end) index ranges
    """
    if timestamps is None or len(timestamps) != n:
        # Simple splitting without time gap detection
        ranges = []
        start = 0
        while start < n:
            end = min(n, start + frag_size)
            ranges.append((start, end))
            if end == n:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return ranges
    
    # Detect segments split by time gaps
    segments = []
    seg_start = 0
    
    for i in range(1, n):
        time_gap = timestamps[i] - timestamps[i-1]
        if time_gap > gap_threshold_ns:
            segments.append((seg_start, i))
            gap_sec = time_gap / 1e9
            print(f"  ⏱ Time gap detected at index {i-1}→{i}: "
                  f"{gap_sec:.2f}s")
            seg_start = i
    
    segments.append((seg_start, n))
    print(f"  Split into {len(segments)} continuous segments "
          f"based on time gaps")
    
    # Split each segment into fragments with overlap
    ranges = []
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        seg_len = seg_end - seg_start
        print(f"  Segment {seg_idx}: indices [{seg_start}:{seg_end}) "
              f"with {seg_len} frames")
        
        start = seg_start
        while start < seg_end:
            end = min(seg_end, start + frag_size)
            ranges.append((start, end))
            if end == seg_end:
                break
            start = end - overlap
            if start < seg_start:
                start = seg_start
            if start >= seg_end:
                break
    
    return ranges


def fuse_fragment(
    colored_clouds: List[o3d.geometry.PointCloud],
    nodes_poses: List[np.ndarray],
    frag_range: Tuple[int, int],
    voxel_merge: float = 0.02
) -> o3d.geometry.PointCloud:
    """
    Transform and merge point clouds within a fragment.
    
    Args:
        colored_clouds: List of all colored point clouds
        nodes_poses: List of poses for clouds in this fragment
        frag_range: Tuple of (start, end) indices
        voxel_merge: Voxel size for final downsampling
        
    Returns:
        Merged and downsampled PointCloud
    """
    frag_start, frag_end = frag_range
    merged = o3d.geometry.PointCloud()
    
    for i in range(frag_start, frag_end):
        pose = nodes_poses[i - frag_start]
        p = copy.deepcopy(colored_clouds[i])
        p.transform(pose)
        merged += p
        
    if voxel_merge and voxel_merge > 0:
        merged = merged.voxel_down_sample(voxel_merge)
        
    return merged


def build_local_fragment(
    colored_clouds: List[o3d.geometry.PointCloud],
    frag_range: Tuple[int, int],
    odom_poses: Optional[List[np.ndarray]] = None,
    voxel_size: float = VOXEL_SIZE,
    max_correspondence_distance: float = MAX_CORRESPONDENCE_DISTANCE,
    verbose: bool = False
) -> Tuple[
    o3d.geometry.PointCloud,
    List[o3d.geometry.PointCloud],
    o3d.pipelines.registration.PoseGraph
]:
    """
    Build a single fragment with local pose graph optimization.
    
    Args:
        colored_clouds: List of all colored point clouds
        frag_range: Tuple of (start, end) indices for this fragment
        odom_poses: Optional list of odometry poses
        voxel_size: Voxel size for preprocessing
        max_correspondence_distance: Max distance for correspondences
        
    Returns:
        Tuple of (fragment_cloud, pcds_std, local_pg)
    """
    frag_start, frag_end = frag_range
    sub = colored_clouds[frag_start:frag_end]
    print(f"\n==== Building fragment [{frag_start}:{frag_end}) "
          f"with {len(sub)} frames ====")
    
    # Preprocess point clouds
    pcds_std = [preprocess_point_cloud(p, voxel_size) for p in sub]
    
    # Extract corresponding odometry poses
    initial_poses = None
    if odom_poses is not None:
        initial_poses = odom_poses[frag_start:frag_end]
    
    # Local registration with colored ICP
    local_pg = full_registration(
        pcds_std, voxel_size, max_correspondence_distance,
        initial_poses, verbose)
    local_pg = optimize_pose_graph(
        local_pg, max_correspondence_distance,
        preference_loop_closure=0.7)
    
    # Extract local poses and fuse fragment
    local_poses = [node.pose for node in local_pg.nodes]
    fragment_cloud = fuse_fragment(
        colored_clouds, local_poses, frag_range, voxel_merge=0.02)
    
    return fragment_cloud, pcds_std, local_pg


def register_fragments(
    fragment_reps: List[o3d.geometry.PointCloud],
    voxel_size: float = VOXEL_SIZE,
    max_correspondence_distance: float = MAX_CORRESPONDENCE_DISTANCE,
    verbose: bool = False
) -> o3d.pipelines.registration.PoseGraph:
    """
    Register fragment representatives into a global pose graph.
    
    Args:
        fragment_reps: List of downsampled fragment point clouds
        voxel_size: Voxel size for processing
        max_correspondence_distance: Max distance for correspondences
        
    Returns:
        Global fragment-level PoseGraph
    """
    print(f"==== Fragment-level registration of {len(fragment_reps)} "
          f"reps (Colored ICP) ====")
    
    pg = o3d.pipelines.registration.PoseGraph()
    odom = np.identity(4)
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(odom))
    
    # Register neighboring fragments
    for i in range(len(fragment_reps) - 1):
        src, tgt = fragment_reps[i], fragment_reps[i+1]
        if verbose:
            print(f"  Frag {i}→{i+1} (colored ICP): ", end="")
        
        src_down = preprocess_point_cloud(src, voxel_size)
        tgt_down = preprocess_point_cloud(tgt, voxel_size)
        
        T_refined, result = colored_icp_registration(
            src_down, tgt_down, voxel_size, max_correspondence_distance)
        
        if T_refined is not None and result is not None:
            fitness = result.fitness
            if not is_valid_transformation(
                    T_refined, max_translation=800.0,
                    max_rotation_deg=600.0):
                if verbose:
                    print("✗ invalid transform → identity edge")
                T_refined = np.eye(4)
                info = np.identity(6) * 0.001
                uncertain = True
            elif fitness < 0.2:
                if verbose:
                    print(f"✗ low fitness={fitness:.3f} → weak edge")
                info = np.identity(6) * 0.1
                uncertain = True
            else:
                if verbose:
                    print(f"✓ fitness={fitness:.3f}, "
                          f"rmse={result.inlier_rmse:.3f}")
                info = np.identity(6) * min(fitness * 2.0, 1.0)
                uncertain = False
        else:
            if verbose:
                print("✗ registration failed → identity edge")
            T_refined = np.eye(4)
            info = np.identity(6) * 0.001
            uncertain = True
        
        odom = T_refined @ odom
        pg.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odom)))
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i, i+1, T_refined, information=info, uncertain=uncertain))
    
    # Add loop closures between non-neighboring fragments
    print("\n  Adding fragment loop closures (colored ICP)...")
    loop_added = 0
    for i in range(0, len(fragment_reps) - 2):
        for j in range(i + 2, len(fragment_reps)):
            src, tgt = fragment_reps[i], fragment_reps[j]
            
            src_down = preprocess_point_cloud(src, voxel_size)
            tgt_down = preprocess_point_cloud(tgt, voxel_size)
            
            T, result = colored_icp_registration(
                src_down, tgt_down, voxel_size, max_correspondence_distance)
            
            if T is not None and result is not None:
                fitness = result.fitness
                if fitness > 0.15 and is_valid_transformation(
                        T, max_translation=1500.0,
                        max_rotation_deg=900.0):
                    info_weight = min(fitness, 0.8)
                    pg.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j, T,
                            information=np.identity(6) * info_weight,
                            uncertain=True))
                    loop_added += 1
                    if verbose:
                        print(f"    Loop {i}→{j}: "
                              f"fitness={fitness:.3f} ✓")
                else:
                    if verbose:
                        print(f"    Loop {i}→{j}: "
                              f"fitness={fitness:.3f} ✗")
            else:
                if verbose:
                    print(f"    Loop {i}→{j}: registration failed ✗")

    print(f"  Added {loop_added} loop closure edges")
    pg = optimize_pose_graph(
        pg, max_correspondence_distance,
        preference_loop_closure=2)
    return pg
