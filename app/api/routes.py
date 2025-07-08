from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import uuid
import cv2
import numpy as np 

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    """Upload user body image for virtual try-on"""
    image_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{image_id}_{image.filename}")
    with open(filepath, "wb") as buffer:
        buffer.write(await image.read())
    return {"message": "Image uploaded successfully", "filepath": filepath}

@router.post("/upload-cloth/")
async def upload_cloth_image(cloth: UploadFile=File(...)):
    """Upload clothing image (top/dress/etc)"""
    cloth_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{cloth_id}_{cloth.filename}")
    with open(filepath, "wb") as buffer:
        buffer.write(await cloth.read())
    return {"message": "Clothing image uploaded successfully", "filepath": filepath}

@router.post("/try-on/")
async def virtual_try_on(
    user_image_path: str = Form(...),
    cloth_image: str = Form(...),
):
    """Apply virtual try-on (dummy version for now).
        Replace this with real DL inference call."""
    try:
        user_image = cv2.imread(user_image_path)
        cloth_image = cv2.imread(cloth_image)
        
        if user_image is None or cloth_image is None:
            return JSONResponse(status_code=400, content={"message": "Invalid image paths"})
        
        h, w, _ = cloth_image.shape
        user_image_path[0:h, 0:w] = cloth_image
        
        # Save result image
        output_path = os.path.join(UPLOAD_DIR, f"tryon_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(output_path, user_image_path)
        
        return {"message": "Virtual try-on successful", "result_filepath": output_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})