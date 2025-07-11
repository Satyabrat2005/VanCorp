from turtle import color
import cv2
import numpy as np
from typing import Optional, Tuple

def read_image(image_path: str, color_image: str) -> Optional[np.ndarray]:
    flag = cv2.IMREAD_COLOR if color == "color" else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(image_path, flag)
    if image is None:
        raise FileNotFoundError(f"❌ Image not found at {image_path}")
    return image

def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size)