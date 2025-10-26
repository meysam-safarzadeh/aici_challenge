"""Point cloud processing, colorization, and preprocessing utilities."""

import numpy as np
import cv2
import open3d as o3d
import struct
from typing import Optional, Tuple


def parse_pointcloud2(msg: any) -> Optional[np.ndarray]:
    """
    Parse ROS2 PointCloud2 message to numpy array.
    
    Args:
        msg: ROS2 PointCloud2 message
        
    Returns:
        numpy array of shape (N, 3) or None if parsing fails
    """
    if not hasattr(msg, 'data') or not hasattr(msg, 'point_step'):
        return None
        
    n_points = len(msg.data) // msg.point_step
    points_list = []
    
    for i in range(n_points):
        offset = i * msg.point_step
        x = struct.unpack_from('f', msg.data, offset)[0]
        y = struct.unpack_from('f', msg.data, offset + 4)[0]
        z = struct.unpack_from('f', msg.data, offset + 8)[0]
        points_list.append([x, y, z])
        
    return np.array(points_list) if points_list else None


def colorize_pointcloud(pts: np.ndarray, img_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, T_cam_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to image and assign colors.
    
    Args:
        pts: (N, 3) numpy array of 3D points in LiDAR frame
        img_bgr: BGR image
        K: 3x3 camera intrinsic matrix
        dist: Distortion coefficients
        T_cam_lidar: 4x4 transform from LiDAR to camera frame
        
    Returns:
        Tuple of (valid_points, colors_rgb) where:
            - valid_points: (M, 3) points that project into image
            - colors_rgb: (M, 3) normalized RGB colors [0, 1]
    """
    h, w = img_bgr.shape[:2]
    
    # Undistort image
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    img_ud = cv2.undistort(img_bgr, K, dist, None, new_K)
    K_use = new_K
    
    # Transform points to camera frame
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_cam_h = (T_cam_lidar @ pts_h.T).T
    pts_cam = pts_cam_h[:, :3]
    
    # Filter points in front of camera
    z = pts_cam[:, 2]
    front_mask = z > 0
    
    # Project to image
    xy = (K_use @ pts_cam[front_mask].T).T
    u = (xy[:, 0] / xy[:, 2]).astype(np.int32)
    v = (xy[:, 1] / xy[:, 2]).astype(np.int32)
    
    # Filter points within image bounds
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid_idx = np.nonzero(front_mask)[0][in_bounds]
    u, v = u[in_bounds], v[in_bounds]
    
    # Extract colors
    colors_bgr = img_ud[v, u, :]
    colors_rgb = colors_bgr[:, ::-1].astype(np.float32) / 255.0
    valid_points = pts[valid_idx]
    
    return valid_points, colors_rgb


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Downsample and estimate normals for point cloud.
    
    Args:
        pcd: Open3D PointCloud
        voxel_size: Voxel size for downsampling
        
    Returns:
        Downsampled PointCloud with normals
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down
