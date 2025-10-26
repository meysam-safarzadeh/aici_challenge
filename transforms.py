"""Coordinate transformation utilities for pose and geometry operations."""

import numpy as np


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        qx, qy, qz, qw: Quaternion components
        
    Returns:
        3x3 rotation matrix
    """
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Compute rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def pose_to_transform_matrix(position: any, orientation: any) -> np.ndarray:
    """
    Convert position and quaternion orientation to 4x4 transform matrix.
    
    Args:
        position: Object with x, y, z attributes
        orientation: Object with x, y, z, w attributes (quaternion)
        
    Returns:
        4x4 transformation matrix
    """
    x, y, z = position.x, position.y, position.z
    qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
    
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    
    return T


def compute_relative_transform(T_current: np.ndarray, T_origin: np.ndarray) -> np.ndarray:
    """
    Compute relative transform from origin to current pose.
    
    Args:
        T_current: Current 4x4 transformation matrix
        T_origin: Origin 4x4 transformation matrix
        
    Returns:
        Relative transformation: T_origin^-1 @ T_current
    """
    T_origin_inv = np.linalg.inv(T_origin)
    T_rel = T_origin_inv @ T_current
    return T_rel


def is_valid_transformation(transformation: np.ndarray, max_translation: float = 2.0, max_rotation_deg: float = 45.0) -> bool:
    """
    Validate transformation for reasonable translation and rotation bounds.
    
    Args:
        transformation: 4x4 transformation matrix
        max_translation: Maximum allowed translation norm
        max_rotation_deg: Maximum allowed rotation in degrees
        
    Returns:
        True if transformation is valid, False otherwise
    """
    translation = np.linalg.norm(transformation[:3, 3])
    if translation > max_translation:
        return False
        
    rotation_matrix = transformation[:3, :3]
    trace = np.trace(rotation_matrix)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle)
    
    if angle_deg > max_rotation_deg:
        return False
        
    return True
