import cv2
from matplotlib.pylab import size
import numpy as np
from typing import Optional

def read_image(imag_path: str, color: str = "color") -> Optional[np.ndarray]:
    """
    Reads an image from the given path.
    """
    flag = cv2.IMREAD_COLOR if color== "color" else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(imag_path, flag)
    if image is None:
        raise FileNotFoundError(f"image not found at {imag_path}")
    return image

def save_image(image: np.ndarray, path: str) -> None:
    """
    Saves an image to disk.

    Args:
        image (np.ndarray): Image to save.
        path (str): Path to save the image.
    """
    cv2.imwrite(path,image)

def resize_image(image: np.ndarray, path: str) -> np.ndarray:
    """
    Resize image to a given (width, height).

    Args:
        image (np.ndarray): Image to resize.
        size (tuple): (width, height)

    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, size) # type: ignore