from fastapi import FastAPI, APIRouter, Form
from fastapi.responses import JSONResponse
from app.ml.tryon_pipeline import run_virtual_tryon 
import uuid
import cv2
import os

router = APIRouter()

@router.post("/")
async def virtual_try_on(
    user_image_path: str = Form(...),
    cloth_image: str = Form(...)
):
    try:
        user_path = f"app/static/uploads/users/{user_image_path}"
        cloth_path = f"app/static/uploads/cloths/{cloth_image}"
        
        result_image = run_virtual_tryon(user_path, cloth_path)
        output_path = f"app/static/outputs/tryon_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(output_path, result_image)

        return{"message" : "virtual try on successful"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"❌ Error: {str(e)}"})

