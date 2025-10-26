# Point cloud concatenation, colorization, and registration

A fragment-based multiway registration system for creating colored 3D point clouds from ROS2 bag data containing LiDAR, RGB camera, and odometry information.

## Approach

This pipeline implements a robust 3D reconstruction approach using fragment-based registration to handle large-scale datasets efficiently:

### 0. Calibration Setup

Before processing the data, we need to establish the geometric relationships between sensors:

#### Camera Intrinsics
The camera intrinsics were extracted from the **P matrix** in the `/zed/zed_node/rgb/camera_info` topic:

```
P matrix (3×4 projection matrix):
┌                                              ┐
│  524.73699951    0.0         649.56481934   0.0  │
│    0.0         524.73699951  368.82150269   0.0  │
│    0.0           0.0           1.0          0.0  │
└                                              ┘
```

This gives us:
- **fx, fy** = 524.737 (focal lengths)
- **cx, cy** = (649.565, 368.822) (principal point)

#### LiDAR to Camera Transform
The extrinsic transform was calculated from the `/tf_static` topic in the ROS2 bag. We compute the transform that maps LiDAR points into the camera optical frame:

$$
\boxed{T_{\text{cam} \leftarrow \text{lidar}} = \left(T_{\text{base} \leftarrow \text{cam}}\right)^{-1} \cdot T_{\text{base} \leftarrow \text{lidar}}}
$$

Where each $T_{\text{parent} \leftarrow \text{child}}$ is a 4×4 homogeneous transform (rotation + translation) that converts coordinates from the child frame into the parent frame.

The resulting transform is hardcoded in `config.py` as `T_CAM_LIDAR`.

### 1. Data Ingestion
- Reads ROS2 bag files containing:
  - **LiDAR data** (`/livox/lidar`) - Point cloud data
  - **RGB images** (`/zed/zed_node/rgb/image_rect_color/compressed`) - Color information
  - **Odometry** (`/odom`) - Initial pose estimates
- Projects RGB colors onto LiDAR points using calibrated camera intrinsics and extrinsics
- Creates colored point clouds with odometry-based poses

### 2. Fragment-Based Processing
Rather than processing all frames at once (which would be memory-intensive and error-prone), the pipeline:
- **Splits data into overlapping fragments** based on:
  - Configurable fragment size (default: 800 frames)
  - Overlap between fragments (default: 400 frames) for continuity
  - Automatic detection of time gaps in the data stream
- **Local registration** within each fragment:
  - Uses ICP (Iterative Closest Point) for pose refinement
  - Builds local pose graphs with consecutive and loop closure edges
  - Optimizes poses within the fragment using pose graph optimization

### 3. Global Registration
- Registers fragment representatives to each other
- Creates a global pose graph connecting all fragments
- Optimizes global poses to ensure consistency across the entire reconstruction

### 4. Final Reconstruction
- Applies optimized global transformations to all fragments
- Merges fragments into a single coherent point cloud
- Performs final voxel downsampling for manageable output size

### Key Features
- **Memory efficient**: Processes large datasets in manageable chunks
- **Robust**: Fragment overlap ensures continuity and loop closure detection
- **Flexible**: Configurable parameters for different scenarios
- **Color preservation**: Maintains RGB information throughout the pipeline

---

## Installation & Usage

### Prerequisites
- Python 3.10 or higher
- Docker (for containerized execution)

### Method 1: Direct CLI Execution

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run with CLI Arguments

**Basic usage** (required argument only):
```bash
python main.py --bag-dir /path/to/rosbag2_directory
```

**With custom parameters**:
```bash
python main.py \
    --bag-dir /path/to/rosbag2_directory \
    --max-samples 50000 \
    --fragment-size 1000 \
    --fragment-overlap 500 \
    --voxel-size 0.05 \
    --min-fitness-consecutive 0.8
```

**All available CLI options**:
```bash
python main.py --help
```

#### Available Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bag-dir` | **Required** | Path to ROS2 bag directory |
| `--max-samples` | 40000 | Maximum number of samples to process |
| `--fragment-size` | 800 | Size of each fragment |
| `--fragment-overlap` | 400 | Overlap between fragments |
| `--voxel-size` | 0.04 | Voxel size for downsampling |
| `--voxel-size-fast` | 0.06 | Voxel size for fast processing |
| `--icp-threshold` | 0.15 | ICP convergence threshold |
| `--loop-closure-interval` | 15 | Interval for loop closure detection |
| `--loop-closure-distance-threshold` | 5.0 | Distance threshold for loop closure |
| `--min-fitness-consecutive` | 0.75 | Minimum fitness for consecutive registration |
| `--min-fitness-loop` | 0.4 | Minimum fitness for loop closure |

---

### Method 2: Docker Execution

#### 1. Build the Docker Image
```bash
docker build -t 3d-reconstruction .
```

#### 2. Run the Container

**Basic usage**:
```bash
docker run \
    -v /path/to/rosbag:/data/rosbag \
    -v $(pwd)/output:/app/output \
    3d-reconstruction --bag-dir /data/rosbag
```

**Example with actual path**:
```bash
docker run \
    -v /home/challenge_aici/office:/data \
    -v $(pwd)/output:/app/output \
    3d-reconstruction --bag-dir /data/rosbag2_2025_10_20-16_09_39
```

**With custom parameters**:
```bash
docker run \
    -v /home/challenge_aici/office:/data \
    -v $(pwd)/output:/app/output \
    3d-reconstruction \
    --bag-dir /data/rosbag2_2025_10_20-16_09_39 \
    --fragment-size 1000 \
    --voxel-size 0.05 \
    --max-samples 50000
```

**With memory limit** (if processing large datasets):
```bash
docker run --memory=8g \
    -v /home/challenge_aici/office:/data \
    -v $(pwd)/output:/app/output \
    3d-reconstruction --bag-dir /data/rosbag2_2025_10_20-16_09_39
```

#### Docker Volume Mounts
- `-v /path/to/rosbag:/data` - Mount your ROS2 bag directory
- `-v $(pwd)/output:/app/output` - Mount output directory to save results locally

---

## Output

The pipeline generates two main output files in the `output/` directory:

1. **`concatenated_colored_cloud.ply`** - Simple concatenation of all colored point clouds
2. **`registered_colored_cloud_fragments.ply`** - Globally registered and optimized point cloud

Additional outputs (if `SAVE_INTERMEDIATE=True` in config):
- `fragments/fragment_XXX.ply` - Individual fragment point clouds

---

## Configuration

Advanced configuration can be modified in `config.py`:
- Camera intrinsics and distortion coefficients
- Extrinsic transform (LiDAR to Camera)
- ROS2 topic names
- Registration parameters
- Output paths

---

## Project Structure

```
.
├── main.py                 # Main pipeline entry point
├── config.py              # Configuration and constants
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── core/
│   ├── fragments.py      # Fragment-based processing
│   ├── io_utils.py       # ROS2 bag reading and I/O
│   ├── pointcloud.py     # Point cloud utilities
│   ├── registration.py   # ICP and pose graph optimization
│   └── transforms.py     # Transformation utilities
├── fragments/            # Intermediate fragment outputs
└── output/              # Final reconstruction outputs
```

---

## Notes

- **Memory**: Processing large datasets may require significant RAM. Consider reducing `--max-samples` or `--fragment-size` for memory-constrained environments.
- **Performance**: Adjust `--voxel-size` to balance quality vs. processing time. Larger values = faster but lower quality.
- **Loop Closures**: Increase `--loop-closure-interval` for faster processing or decrease for better accuracy in environments with many revisits.

---

## License

[Add your license information here]
