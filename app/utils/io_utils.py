import os 
import uuid
from typing import Tuple

def get_unique_filename(filename: str) -> str:
    ext = os.path.splitext(filename)[1]
    return f"{uuid.uuid4().hex}{ext}"


def get_user_image_path(filename: str) -> str:
    return os.path.join("app", "static", "uploads", "users", filename)

def get_cloth_image_path(filename: str) -> str:
    return os.path.join("app", "static", "uploads", "cloths", filename)

def get_output_path(filename: str) -> str:
    return os.path.join("app", "static", "outputs", filename)

def validate_image_extension(filename: str) -> bool:
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))