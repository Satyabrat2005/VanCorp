import cv2
import numpy as np

def wrap_cloth(cloth_image: np.ndarray, keypoints: dict, target_shape: tuple) -> np.ndarray:
    """
    Applies affine transformation to warp the segmented cloth 
    based on user's keypoints (shoulders and hips).

    Args:
        cloth_image (np.ndarray): The segmented cloth image.
        keypoints (dict): Keypoints detected on the user.
        target_shape (tuple): Shape of the user image (H, W, C).

    Returns:
        np.ndarray: Warped cloth image aligned to user body.
    """
    h,w,_ = target_shape
    src_pts = np.float32([
        [0, 0],
        [cloth_image.shape[1], 0],
        [0, cloth_image.shape[0]]
    ]) # type: ignore
    
    dst_pts = np.float32([
        keypoints["left_shoulder"],
        keypoints["right_shoulder"],
        keypoints["left_hip"]
    ])  # type: ignore

    M = cv2.getAffineTransform(src_pts, dst_pts) # type: ignore

    # Warp cloth
    warped = cv2.warpAffine(cloth_image, M, (w, h))

    return warped

