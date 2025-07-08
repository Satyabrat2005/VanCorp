import cv2
import numpy as np
from PIL import Image
def read_image(path):
    return cv2.imread(path)

def save_image(path, image):
    cv2.imwrite(path, image)

def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))