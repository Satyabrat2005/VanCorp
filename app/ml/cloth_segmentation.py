import cv2
import numpy as np

def segment_clothes(cloth_image: np.ndarray) -> np.ndarray:
    """
    Dummy cloth segmentation (HSV thresholding).
    Replace with DeepLab or CIHP parsing later.
    """
    hsv = cv2.cvtColor(cloth_image,  cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    segmented = cv2.bitwise_and(cloth_image, cloth_image, mask=mask)
    return segmented
