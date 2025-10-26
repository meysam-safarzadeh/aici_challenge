import copy
import argparse
from pathlib import Path
import open3d as o3d
from config import *
from core.io_utils import read_rosbag, create_colored_clouds, save_point_cloud
from core.fragments import (split_into_fragments, build_local_fragment, 
                       register_fragments)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fragment-based multiway registration for 3D reconstruction')
    
    # Required argument
    parser.add_argument('--bag-dir', type=str, required=True,
                        help='Path to ROS2 bag directory (required)')
    
    # Optional arguments with defaults from config.py
    parser.add_argument('--max-samples', type=int, default=40000,
                        help='Maximum number of samples to process (default: 40000)')
    parser.add_argument('--fragment-size', type=int, default=800,
                        help='Size of each fragment (default: 800)')
    parser.add_argument('--fragment-overlap', type=int, default=400,
                        help='Overlap between fragments (default: 400)')
    parser.add_argument('--voxel-size', type=float, default=0.04,
                        help='Voxel size for downsampling (default: 0.04)')
    parser.add_argument('--voxel-size-fast', type=float, default=0.06,
                        help='Voxel size for fast processing (default: 0.06)')
    parser.add_argument('--icp-threshold', type=float, default=0.15,
                        help='ICP convergence threshold (default: 0.15)')
    parser.add_argument('--loop-closure-interval', type=int, default=15,
                        help='Interval for loop closure detection (default: 15)')
    parser.add_argument('--loop-closure-distance-threshold', type=float, default=5.0,
                        help='Distance threshold for loop closure (default: 5.0)')
    parser.add_argument('--min-fitness-consecutive', type=float, default=0.75,
                        help='Minimum fitness for consecutive registration (default: 0.75)')
    parser.add_argument('--min-fitness-loop', type=float, default=0.4,
                        help='Minimum fitness for loop closure (default: 0.4)')
    
    return parser.parse_args()


def main() -> None:
    """Main reconstruction pipeline."""
    # Parse command-line arguments
    args = parse_args()
    
    # Override config values with CLI arguments
    BAG_DIR_CLI = Path(args.bag_dir)
    MAX_SAMPLES_CLI = args.max_samples
    FRAGMENT_SIZE_CLI = args.fragment_size
    FRAGMENT_OVERLAP_CLI = args.fragment_overlap
    VOXEL_SIZE_CLI = args.voxel_size
    VOXEL_SIZE_FAST_CLI = args.voxel_size_fast
    ICP_THRESHOLD_CLI = args.icp_threshold
    LOOP_CLOSURE_INTERVAL_CLI = args.loop_closure_interval
    LOOP_CLOSURE_DISTANCE_THRESHOLD_CLI = args.loop_closure_distance_threshold
    MIN_FITNESS_CONSECUTIVE_CLI = args.min_fitness_consecutive
    MIN_FITNESS_LOOP_CLI = args.min_fitness_loop
    MAX_CORRESPONDENCE_DISTANCE_CLI = VOXEL_SIZE_CLI * 3.5
    
    print("=" * 70)
    print("FRAGMENT-BASED MULTIWAY REGISTRATION")
    print("=" * 70)
    
    # Ensure output directories exist
    ensure_output_dirs()
    
    # Step 1: Read ROS2 bag data
    rgb_data, lidar_data, odom_data = read_rosbag(
        BAG_DIR_CLI, RGB_TOPIC, LIDAR_TOPIC, ODOM_TOPIC, MAX_SAMPLES_CLI)
    
    # Step 2: Create colored point clouds with odometry
    colored_clouds, odom_poses, cloud_timestamps = create_colored_clouds(
        rgb_data, lidar_data, odom_data, K, DIST_COEFFS, T_CAM_LIDAR)
    
    N = len(colored_clouds)
    if N == 0:
        raise SystemExit("No valid colored clouds created.")
    
    # step 3: Concatenate point clouds and save
    print(f"\nConcatenating {len(colored_clouds)} colored point clouds...")
    concatenated_point_cloud = copy.deepcopy(colored_clouds[0])
    for colored_point_cloud in colored_clouds[1:]:
        concatenated_point_cloud += colored_point_cloud
    
    # Save the final concatenated point cloud
    o3d.io.write_point_cloud(OUTPUT_CONCAT_PLY, concatenated_point_cloud)
    print(f"\n✓ Saved concatenated colored point cloud to '{OUTPUT_CONCAT_PLY}'")
    print(f"  Total points: {len(concatenated_point_cloud.points)}")

    # Step 4: Split into fragments
    print("\n" + "=" * 70)
    print("Splitting into fragments based on time gaps and size")
    print("=" * 70)
    frag_ranges = split_into_fragments(
        N, FRAGMENT_SIZE_CLI, FRAGMENT_OVERLAP_CLI, cloud_timestamps,
        gap_threshold_ns=1e9)
    print(f"\nPlanned {len(frag_ranges)} fragments: {frag_ranges}")
    
    # Step 5: Build local fragments
    fragment_clouds = []
    fragment_representatives = []
    local_pose_graphs = []
    
    for frag_id, fr in enumerate(frag_ranges):
        frag_cloud, pcds_std, local_pg = build_local_fragment(
            colored_clouds, fr, odom_poses, VOXEL_SIZE_CLI,
            MAX_CORRESPONDENCE_DISTANCE_CLI, VERBOSE)
        local_pose_graphs.append(local_pg)
        
        # Save intermediate fragment
        if SAVE_INTERMEDIATE:
            frag_path = FRAG_DIR / f"fragment_{frag_id:03d}.ply"
            save_point_cloud(frag_cloud, frag_path)
        
        fragment_clouds.append(frag_cloud)
        
        # Create downsampled representative for fragment registration
        rep = frag_cloud.voxel_down_sample(
            max(VOXEL_SIZE_CLI * 2, 0.05))
        fragment_representatives.append(rep)
    
    # Step 6: Register fragments globally
    frag_pg = register_fragments(
        fragment_representatives, VOXEL_SIZE_CLI,
        MAX_CORRESPONDENCE_DISTANCE_CLI, VERBOSE)
    frag_global_poses = [node.pose for node in frag_pg.nodes]
    
    # Step 7: Merge fragments into final point cloud
    print("\n" + "=" * 70)
    print("Applying fragment global poses and merging")
    print("=" * 70)
    
    final = o3d.geometry.PointCloud()
    for i, frag_cloud in enumerate(fragment_clouds):
        Tglob = frag_global_poses[i]
        c = copy.deepcopy(frag_cloud)
        c.transform(Tglob)
        final += c
    
    # Final downsampling for manageable file size
    final = final.voxel_down_sample(0.01)
    
    # Save final result
    save_point_cloud(final, OUTPUT_PLY)
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"✓ Saved global registered colored point cloud to "
          f"'{OUTPUT_PLY}'")
    print(f"  #Fragments: {len(fragment_clouds)}  |  "
          f"Total frames: {N}")
    print(f"  Total points (after final downsample): "
          f"{len(final.points)}")


if __name__ == "__main__":
    main()
