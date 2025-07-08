import cv2
import numpy as np
import os
from app.ml.cloth_segmentation import segment_clothes  # type: ignore
from app.ml.pose_estimator import estimate_pose        # type: ignore

def run_virtual_tryon(user_image_path: str, cloth_image_path: str) -> np.ndarray:
    """
    Run the full virtual try-on pipeline: cloth segmentation, pose estimation, and overlay.

    Args:
        user_image_path (str): Path to the uploaded user image.
        cloth_image_path (str): Path to the selected clothing image.

    Returns:
        np.ndarray: The output image after try-on.
    """

    # Resolve full absolute paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    user_image_path = os.path.join(ROOT_DIR, "app", "static", "uploads", "users", os.path.basename(user_image_path))
    cloth_image_path = os.path.join(ROOT_DIR, "app", "static", "uploads", "cloths", os.path.basename(cloth_image_path))

    print("📸 USER IMAGE PATH:", user_image_path)
    print("👕 CLOTH IMAGE PATH:", cloth_image_path)

    # Load images
    user_image = cv2.imread(user_image_path)
    cloth_image = cv2.imread(cloth_image_path)

    if user_image is None:
        raise ValueError(f"❌ Failed to load user image from path: {user_image_path}")
    if cloth_image is None:
        raise ValueError(f"❌ Failed to load cloth image from path: {cloth_image_path}")

    print("✅ Both images loaded successfully.")

    # Step 1: Segment the clothing item
    print("🔍 Segmenting cloth...")
    segmented_cloth = segment_clothes(cloth_image) # type: ignore
    print("✅ Cloth segmented.")

    # Step 2: Estimate pose on the user image
    print("🧍 Estimating user pose...")
    keypoints = estimate_pose(user_image)
    print("✅ Pose estimated.")

    # Step 3: Overlay the clothing on the user
    print("🎨 Overlaying segmented cloth on user...")
    tryon_result = overlay_cloth_on_user(user_image, segmented_cloth, keypoints) # type: ignore
    print("✅ Virtual try-on complete.")

    return tryon_result


def overlay_cloth_on_user(user_image: np.ndarray, cloth_image: np.ndarray, keypoints: dict) -> np.ndarray:
    """
    Warp and overlay the cloth image onto the user's torso area using affine transformation.
    """

    # Get required keypoints
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")
    left_hip = keypoints.get("left_hip")

    if not all([left_shoulder, right_shoulder, left_hip]):
        raise ValueError("❌ Required keypoints (shoulders, left_hip) missing for warping.")

    # Step 1: Prepare 3 source points from the cloth image corners
    h, w = cloth_image.shape[:2]
    src_pts = np.float32([
        [0, 0],          # Top-left of cloth
        [w, 0],          # Top-right of cloth
        [0, h]           # Bottom-left of cloth
    ]) # type: ignore

    # Step 2: Target 3 destination keypoints on user's torso
    dst_pts = np.float32([
        left_shoulder,
        right_shoulder,
        left_hip
    ]) # type: ignore

    # Step 3: Get affine transform matrix and warp the cloth image
    M = cv2.getAffineTransform(src_pts, dst_pts) # type: ignore
    warped_cloth = cv2.warpAffine(cloth_image, M, (user_image.shape[1], user_image.shape[0]))

    # Step 4: Create a binary mask from warped cloth to blend only non-background parts
    gray = cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    alpha_inv = cv2.bitwise_not(alpha_3)

    # Step 5: Combine user and warped cloth using masks
    user_bg = cv2.bitwise_and(user_image, alpha_inv)
    cloth_fg = cv2.bitwise_and(warped_cloth, alpha_3)
    combined = cv2.add(user_bg, cloth_fg)

    return combined