
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import os
import uuid

router = APIRouter()

UPLOAD_DIR_USER = "app/static/uploads/users"
UPLOAD_DIR_CLOTH = "app/static/uploads/cloths"
os.makedirs(UPLOAD_DIR_USER, exist_ok=True)
os.makedirs(UPLOAD_DIR_CLOTH, exist_ok=True)

@router.post("/user/")
async def upload_user_image(image: UploadFile = File(...)):
    """
    Upload user image and save to user uploads directory.
    """
    unique_name = f"{uuid.uuid4().hex}_{image.filename}"
    filepath = os.path.join(UPLOAD_DIR_USER, unique_name)

    with open(filepath, "wb") as f:
        f.write(await image.read())

    return JSONResponse(content={
        "message": "✅ User image uploaded",
        "filename": unique_name,
        "path": filepath
    })


@router.post("/cloth/")
async def upload_cloth_image(cloth: UploadFile = File(...)):
    """
    Upload clothing image and save to cloth uploads directory.
    """
    unique_name = f"{uuid.uuid4().hex}_{cloth.filename}"
    filepath = os.path.join(UPLOAD_DIR_CLOTH, unique_name)

    with open(filepath, "wb") as f:
        f.write(await cloth.read())

    return JSONResponse(content={
        "message": "✅ Cloth image uploaded",
        "filename": unique_name,
        "path": filepath
    })
