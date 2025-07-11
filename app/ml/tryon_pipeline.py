import cv2
import numpy as np
import os

from app.ml.cloth_segmentation import segment_clothes
from app.ml.pose_estimator import estimate_pose
from app.ml.cloth_wrapping import wrap_cloth
from app.ml.fusion import fuse_tryon_output

def run_virtual_tryon(user_image_path: str, cloth_image_path: str):
    """
    Full end-to-end virtual try-on pipeline:
    1. Load images
    2. Segment cloth
    3. Estimate user pose
    4. Warp cloth
    5. Fuse & render output

    Returns:
        np.ndarray: final try-on result
    """

    user_image = cv2.imread(user_image_path)
    cloth_image = cv2.imread(cloth_image_path)

    if user_image is None:
        raise FileNotFoundError(f"❌ User image not found at: {user_image_path}")
    if cloth_image is None:
        raise FileNotFoundError(f"❌ Cloth image not found at: {cloth_image_path}")
    
    print("images loaded.")

    segmented_cloth = segment_clothes(cloth_image)
    print("🧵 Cloth segmented.")

    # 3. Estimate user pose
    keypoints = estimate_pose(user_image)
    print("🧍 User pose estimated.")

    # 4. Warp cloth to fit body
    warped_cloth = wrap_cloth(segmented_cloth, keypoints, user_image.shape)
    print("🪄 Cloth warped to match user.")

    # 5. Fuse (final render)
    final_output = fuse_tryon_output(user_image, warped_cloth, keypoints)
    print("🎨 Final try-on image generated.")

    return final_output