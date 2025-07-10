import cv2
import numpy as np

def fuse_tryon_output(user_image: np.ndarray, cloth_image: np.ndarray, keywords:dict) -> np.ndarray:
    """
    Combines the warped cloth with user image using basic alpha blending.

    Args:
        user_image (np.ndarray): Original user image.
        warped_cloth (np.ndarray): Warped clothing image.
        keypoints (dict): Pose landmarks.

    Returns:
        np.ndarray: Final output image.
    """

    gray = cv2.cvtColor(wrap_cloth, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    mask_rgb = cv2.merge([mask, mask, mask])
    mask_inv_rgb = cv2.merge([mask_inv, mask_inv, mask_inv])

    user_bg = cv2.bitwise_and(user_image, mask_inv_rgb)
    cloth_fg = cv2.bitwise_and(wrap_cloth, mask_rgb)

    blended = cv2.add(user_bg, cloth_fg)
    return blended