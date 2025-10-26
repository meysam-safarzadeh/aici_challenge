"""Configuration and constants for the 3D reconstruction pipeline."""

from pathlib import Path
import numpy as np
from typing import Dict


# ==== Data Sources ====
BAG_DIR = Path("/home/challenge_aici/office/rosbag2_2025_10_20-16_09_39")
RGB_TOPIC = "/zed/zed_node/rgb/image_rect_color/compressed"
LIDAR_TOPIC = "/livox/lidar"
ODOM_TOPIC = "/odom"
MAX_SAMPLES = 40000


# ==== Fragmenting Parameters ====
FRAGMENT_SIZE = 800
FRAGMENT_OVERLAP = 400
SAVE_INTERMEDIATE = True


# ==== Output Configuration ====
OUTPUT_DIR = Path("output")
OUTPUT_PLY = OUTPUT_DIR / "registered_colored_cloud_fragments.ply"
OUTPUT_CONCAT_PLY = OUTPUT_DIR / "concatenated_colored_cloud.ply"
VERBOSE = False
FRAG_DIR = OUTPUT_DIR / "fragments"


# ==== Registration Parameters ====
VOXEL_SIZE = 0.04
VOXEL_SIZE_FAST = 0.06
ICP_THRESHOLD = 0.15
MAX_CORRESPONDENCE_DISTANCE = VOXEL_SIZE * 3.5

LOOP_CLOSURE_INTERVAL = 15
LOOP_CLOSURE_DISTANCE_THRESHOLD = 5

MIN_FITNESS_CONSECUTIVE = 0.75
MIN_FITNESS_LOOP = 0.4


# ==== Camera Intrinsics (Pinhole Model) ====
CAMERA_INTRINSICS: Dict[str, float] = {
    'fx': 524.73699951,
    'fy': 524.73699951,
    'cx': 649.56481934,
    'cy': 368.82150269,
}

K = np.array([
    [CAMERA_INTRINSICS['fx'], 0, CAMERA_INTRINSICS['cx']],
    [0, CAMERA_INTRINSICS['fy'], CAMERA_INTRINSICS['cy']],
    [0, 0, 1]
], dtype=np.float64)

# Distortion coefficients (k1, k2, p1, p2, k3)
DIST_COEFFS = np.zeros(5, dtype=np.float64)


# ==== Extrinsic Transform: LiDAR to Camera ====
T_CAM_LIDAR = np.array([
    [0.0, -1.0,  0.0,  0.060   ],
    [0.0,  0.0, -1.0, -0.020   ],
    [1.0,  0.0,  0.0, -0.088609],
    [0.0,  0.0,  0.0,  1.0     ],
], dtype=float)


def ensure_output_dirs() -> None:
    """Create necessary output directories."""
    FRAG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
