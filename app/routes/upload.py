from fastapi import APIRouter, File, FastAPI, UploadFile
from fastapi.responses import JSONResponse
import os
import uuid

router = APIRouter()

UPLOAD_DIR_USER = "app/static/uploads/users"
UPLOAD_DIR_CLOTH = "app/static/uploads/cloths"
os.makedirs(UPLOAD_DIR_USER, exist_ok=True)
os.makedirs(UPLOAD_DIR_CLOTH, exist_ok=True)

@router.post("/image")
async def upload_user_image(image: UploadFile = File(...)):
    image_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR_USER)
    with open(filepath, "wb") as f:
        f.write(await image.read())
    return {"message": "✅ User image uploaded", "filepath": os.path.basename(filepath)}

@router.post("/cloth")
async def upload_cloth_image(cloth: UploadFile = File(...)):
    cloth_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR_CLOTH, f"{cloth_id}_{cloth.filename}")
    with open(filepath, "wb") as f:
        f.write(await cloth.read())
    return {"message": "✅ Cloth image uploaded", "filepath": os.path.basename(filepath)}