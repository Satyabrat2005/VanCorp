import cv2
import numpy as np
from typing import Dict, Tuple
from app.ml.tps_wrap import TPSWarp

def wrap_cloth(
    cloth_image: np.ndarray,
    keypoints: Dict[str, Tuple[int, int]],
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Warps the segmented cloth image to fit user body using Thin Plate Spline (TPS),
    with handling for different pose modes (standing, sitting, turning).

    Args:
        cloth_image (np.ndarray): Segmented cloth image.
        keypoints (dict): Body keypoints from pose estimator.
        target_shape (tuple): Shape of the user image (H, W, C).

    Returns:
        np.ndarray: Warped cloth image aligned with user.
    """
    h, w, _ = target_shape
    tps = TPSWarp()

    # Basic source grid points on the cloth image (corners)
    src_pts = np.array([
        [0, 0],
        [cloth_image.shape[1], 0],
        [0, cloth_image.shape[0]],
        [cloth_image.shape[1], cloth_image.shape[0]]
    ], dtype=np.float32)

    # Pose-aware target points
    pose_mode = keypoints.get("pose_mode", "standing")

    if pose_mode == "standing":
        dst_pts = np.array([
            keypoints["left_shoulder"],
            keypoints["right_shoulder"],
            keypoints["left_hip"],
            keypoints["right_hip"]
        ], dtype=np.float32)
    
    elif pose_mode == "sitting":
        # Lower the hips slightly to simulate sitting posture
        dst_pts = np.array([
            keypoints["left_shoulder"],
            keypoints["right_shoulder"],
            (keypoints["left_hip"][0], keypoints["left_hip"][1] + 20),
            (keypoints["right_hip"][0], keypoints["right_hip"][1] + 20)
        ], dtype=np.float32)

    elif pose_mode == "turning":
        # Compress horizontal spread to simulate turning
        mid_shoulder_x = (keypoints["left_shoulder"][0] + keypoints["right_shoulder"][0]) // 2
        mid_hip_x = (keypoints["left_hip"][0] + keypoints["right_hip"][0]) // 2
        dst_pts = np.array([
            (mid_shoulder_x - 10, keypoints["left_shoulder"][1]),
            (mid_shoulder_x + 10, keypoints["right_shoulder"][1]),
            (mid_hip_x - 10, keypoints["left_hip"][1]),
            (mid_hip_x + 10, keypoints["right_hip"][1])
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown pose_mode: {pose_mode}")

    # TPS warp
    warped = tps.warp_image(cloth_image, src_pts, dst_pts, (w, h))
    return warped
