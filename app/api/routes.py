from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import uuid
import cv2
import numpy as np
from app.ml.tryon_pipelines import run_virtual_tryon  # type: ignore

router = APIRouter()

# Directory setup
USER_UPLOAD_DIR = "app/static/uploads/users"
CLOTH_UPLOAD_DIR = "app/static/uploads/cloths"
OUTPUT_DIR = "app/static/outputs"

os.makedirs(USER_UPLOAD_DIR, exist_ok=True)
os.makedirs(CLOTH_UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    """Upload user body image for virtual try-on"""
    filename = image.filename
    filepath = os.path.join(USER_UPLOAD_DIR, filename) # type: ignore

    with open(filepath, "wb") as buffer:
        buffer.write(await image.read())

    return {
        "message": "✅ User image uploaded successfully",
        "filepath": filename  # Just filename, not full path
    }


@router.post("/upload-cloth/")
async def upload_cloth_image(cloth: UploadFile = File(...)):
    """Upload clothing image (top/dress/etc)"""
    filename = cloth.filename
    filepath = os.path.join(CLOTH_UPLOAD_DIR, filename) # type: ignore

    with open(filepath, "wb") as buffer:
        buffer.write(await cloth.read())

    return {
        "message": "✅ Clothing image uploaded successfully",
        "filepath": filename
    }


@router.post("/try-on/")
async def virtual_try_on(
    user_image_path: str = Form(..., description="Full filepath from /upload-image/"),
    cloth_image: str = Form(..., description="Full filepath from /upload-cloth/"),
):
    try:
        # Resolve paths from uploaded filenames
        user_path = user_image_path
        cloth_path = cloth_image

        result_image = run_virtual_tryon(user_path, cloth_path)

        # Save output image
        result_filename = f"tryon_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, result_filename)
        cv2.imwrite(output_path, result_image)

        return {
            "message": "✅ Virtual try-on successful!",
            "result_filepath": f"static/outputs/{result_filename}"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"❌ An error occurred: {str(e)}"}
        )
