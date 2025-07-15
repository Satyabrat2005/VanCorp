import cv2
import numpy as np
import os

from app.ml.cloth_segmentation import segment_clothes
from app.ml.pose_estimator import estimate_pose
from app.ml.cloth_wrapping import wrap_cloth
from app.ml.fusion import fuse_tryon_output

def run_virtual_tryon(user_image_path: str, cloth_image_path: str, use_hugging_face: bool = False, pose_mode: str = "standing") -> np.ndarray:
    """
    Full end-to-end virtual try-on pipeline:
    1. Load images
    2. Segment cloth
    3. Estimate user pose
    4. Warp cloth
    5. Fuse & render output

    Args:
        user_image_path (str): Path to user image
        cloth_image_path (str): Path to clothing image
        use_hugging_face (bool): Use Hugging Face diffusion for fusion
        pose_mode (str): One of ['standing', 'sitting', 'turning']

    Returns:
        np.ndarray: Final try-on result
    """

    # 1. Load images
    user_image = cv2.imread(user_image_path)
    cloth_image = cv2.imread(cloth_image_path)

    if user_image is None:
        raise FileNotFoundError(f"❌ User image not found at: {user_image_path}")
    if cloth_image is None:
        raise FileNotFoundError(f"❌ Cloth image not found at: {cloth_image_path}")
    
    print("🖼️ Images loaded.")

    # 2. Segment cloth
    segmented_cloth = segment_clothes(cloth_image)
    print("🧵 Cloth segmented.")

    # 3. Estimate user pose
    keypoints = estimate_pose(user_image)
    keypoints["pose_mode"] = pose_mode  # type: ignore # 🔥 inject mode into keypoints
    print(f"🧍 User pose estimated. Mode: {pose_mode}")

    # 4. Warp cloth
    warped_cloth = wrap_cloth(segmented_cloth, keypoints, user_image.shape) # type: ignore
    print("🪄 Cloth warped.")

    # 5. Fuse with HuggingFace or OpenCV alpha blending
    final_output = fuse_tryon_output(
        user_image=user_image,
        cloth_image=warped_cloth,
        keypoints=keypoints,
        use_diffusers=use_hugging_face
    )
    print("🎨 Final try-on image generated.")

    return final_output
