import copy
import open3d as o3d
from config import *
from core.io_utils import read_rosbag, create_colored_clouds, save_point_cloud
from core.fragments import (split_into_fragments, build_local_fragment, 
                       register_fragments)


def main() -> None:
    """Main reconstruction pipeline."""
    print("=" * 70)
    print("FRAGMENT-BASED MULTIWAY REGISTRATION")
    print("=" * 70)
    
    # Ensure output directories exist
    ensure_output_dirs()
    
    # Step 1: Read ROS2 bag data
    rgb_data, lidar_data, odom_data = read_rosbag(
        BAG_DIR, RGB_TOPIC, LIDAR_TOPIC, ODOM_TOPIC, MAX_SAMPLES)
    
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
        N, FRAGMENT_SIZE, FRAGMENT_OVERLAP, cloud_timestamps,
        gap_threshold_ns=1e9)
    print(f"\nPlanned {len(frag_ranges)} fragments: {frag_ranges}")
    
    # Step 5: Build local fragments
    fragment_clouds = []
    fragment_representatives = []
    local_pose_graphs = []
    
    for frag_id, fr in enumerate(frag_ranges):
        frag_cloud, pcds_std, local_pg = build_local_fragment(
            colored_clouds, fr, odom_poses, VOXEL_SIZE,
            MAX_CORRESPONDENCE_DISTANCE, VERBOSE)
        local_pose_graphs.append(local_pg)
        
        # Save intermediate fragment
        if SAVE_INTERMEDIATE:
            frag_path = FRAG_DIR / f"fragment_{frag_id:03d}.ply"
            save_point_cloud(frag_cloud, frag_path)
        
        fragment_clouds.append(frag_cloud)
        
        # Create downsampled representative for fragment registration
        rep = frag_cloud.voxel_down_sample(
            max(VOXEL_SIZE * 2, 0.05))
        fragment_representatives.append(rep)
    
    # Step 6: Register fragments globally
    frag_pg = register_fragments(
        fragment_representatives, VOXEL_SIZE,
        MAX_CORRESPONDENCE_DISTANCE, VERBOSE)
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
