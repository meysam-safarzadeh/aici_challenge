"""I/O utilities for reading ROS bags and writing point clouds."""

import cv2
import numpy as np
import open3d as o3d
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from core.pointcloud import parse_pointcloud2, colorize_pointcloud
from core.transforms import pose_to_transform_matrix, compute_relative_transform


def find_closest_timestamp(
    target_t: int,
    timestamp_dict: Dict[int, any]
) -> Optional[int]:
    """
    Find the closest timestamp in a dictionary.
    
    Args:
        target_t: Target timestamp
        timestamp_dict: Dictionary with timestamps as keys
        
    Returns:
        Closest timestamp or None if dictionary is empty
    """
    if not timestamp_dict:
        return None
    closest_t = min(timestamp_dict.keys(), key=lambda t: abs(t - target_t))
    return closest_t


def read_rosbag(
    bag_dir: Path,
    rgb_topic: str,
    lidar_topic: str,
    odom_topic: str,
    max_samples: int
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
           Dict[int, np.ndarray]]:
    """
    Read RGB images, LiDAR point clouds, and odometry from ROS2 bag.
    
    Args:
        bag_dir: Path to bag directory
        rgb_topic: RGB image topic name
        lidar_topic: LiDAR topic name
        odom_topic: Odometry topic name
        max_samples: Maximum number of samples to read
        
    Returns:
        Tuple of (rgb_data, lidar_data, odom_data) as dictionaries
        keyed by timestamp
    """
    print("\nReading ROS2 bag...")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    rgb_data = {}
    lidar_data = {}
    odom_data = {}
    
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        rgb_conns = [c for c in reader.connections if c.topic == rgb_topic]
        lidar_conns = [c for c in reader.connections if c.topic == lidar_topic]
        odom_conns = [c for c in reader.connections if c.topic == odom_topic]
        
        if not rgb_conns:
            print(f"Warning: Topic {rgb_topic!r} not found in bag.")
        if not lidar_conns:
            print(f"Warning: Topic {lidar_topic!r} not found in bag.")
        if not odom_conns:
            print(f"Warning: Topic {odom_topic!r} not found in bag.")
            
        all_conns = rgb_conns + lidar_conns + odom_conns
        if not all_conns:
            raise SystemExit("No topics found in bag.")
            
        print("Extracting messages from bag...")
        for conn, t, raw in reader.messages(connections=all_conns):
            if len(lidar_data) >= max_samples:
                print(f"Reached maximum of {max_samples} point clouds, "
                      f"stopping extraction...")
                break
                
            if conn.topic == rgb_topic:
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    rgb_data[t] = img_bgr
                    if len(rgb_data) % 100 == 0:
                        print(f"  RGB frames: {len(rgb_data)}")
                        
            elif conn.topic == lidar_topic:
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                points = parse_pointcloud2(msg)
                if points is not None:
                    lidar_data[t] = points
                    if len(lidar_data) % 100 == 0:
                        print(f"  Point clouds: {len(lidar_data)}")
                        
            elif conn.topic == odom_topic:
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                pose = msg.pose.pose
                T = pose_to_transform_matrix(pose.position, pose.orientation)
                odom_data[t] = T
                if len(odom_data) % 200 == 0:
                    print(f"  Odometry messages: {len(odom_data)}")
    
    print(f"\nCollected {len(rgb_data)} RGB frames, "
          f"{len(lidar_data)} point clouds, "
          f"and {len(odom_data)} odometry messages")
    
    return rgb_data, lidar_data, odom_data


def create_colored_clouds(
    rgb_data: Dict[int, np.ndarray],
    lidar_data: Dict[int, np.ndarray],
    odom_data: Dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    T_cam_lidar: np.ndarray
) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray], List[int]]:
    """
    Create colored point clouds from RGB and LiDAR data with odometry.
    
    Args:
        rgb_data: Dictionary of RGB images keyed by timestamp
        lidar_data: Dictionary of point clouds keyed by timestamp
        odom_data: Dictionary of odometry transforms keyed by timestamp
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        T_cam_lidar: Transform from LiDAR to camera
        
    Returns:
        Tuple of (colored_clouds, odom_poses, cloud_timestamps)
    """
    print("\n" + "=" * 70)
    print("Creating colored point clouds with odometry")
    print("=" * 70)
    
    colored_clouds = []
    odom_poses = []
    cloud_timestamps = []
    first_odom_t = None
    T_origin = None
    
    for idx, (rgb_t, img_bgr) in enumerate(sorted(rgb_data.items())):
        closest_lidar_t = find_closest_timestamp(rgb_t, lidar_data)
        if closest_lidar_t is None:
            continue
        
        closest_odom_t = find_closest_timestamp(rgb_t, odom_data)
        if closest_odom_t is None:
            print(f"Skipping RGB frame at t={rgb_t}: "
                  f"no odometry available")
            continue
        
        # Set origin pose from first odometry
        if T_origin is None:
            T_origin = odom_data[closest_odom_t]
            first_odom_t = closest_odom_t
            print(f"Set origin pose from odometry at t={first_odom_t}")
        
        points = lidar_data[closest_lidar_t]
        valid_points, colors = colorize_pointcloud(
            points, img_bgr, K, dist, T_cam_lidar)
        
        if len(valid_points) > 100:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Transform by relative odometry
            T_current = odom_data[closest_odom_t]
            T_rel = compute_relative_transform(T_current, T_origin)
            pcd.transform(T_rel)
            
            colored_clouds.append(pcd)
            odom_poses.append(T_rel)
            cloud_timestamps.append(rgb_t)
            
            if (idx + 1) % 200 == 0:
                print(f"  Created {len(colored_clouds)} "
                      f"colored point clouds...")
    
    print(f"\n✓ Created {len(colored_clouds)} colored point clouds "
          f"with initial odometry poses")
    return colored_clouds, odom_poses, cloud_timestamps


def save_point_cloud(
    pcd: o3d.geometry.PointCloud,
    filepath: Path
) -> None:
    """
    Save point cloud to PLY file.
    
    Args:
        pcd: Open3D PointCloud
        filepath: Output file path
    """
    o3d.io.write_point_cloud(str(filepath), pcd)
    print(f"  ⬇ Saved point cloud to {filepath}")
