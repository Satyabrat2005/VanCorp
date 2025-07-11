
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from app.ml.tryon_pipeline import run_virtual_tryon
import uuid
import cv2
import os

router = APIRouter()

OUTPUT_DIR = "app/static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.post("/")
async def virtual_try_on(
    user_image_path: str = Form(...),
    cloth_image: str = Form(...)
):
    """
    Perform virtual try-on with given user and cloth image filenames.
    """
    try:
        user_path = os.path.join("app", "static", "uploads", "users", user_image_path)
        cloth_path = os.path.join("app", "static", "uploads", "cloths", cloth_image)

        result_image = run_virtual_tryon(user_path, cloth_path)

        output_filename = f"tryon_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, result_image)

        return {
            "message": "✅ Virtual try-on successful!",
            "output_image": output_filename,
            "output_path": output_path
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"❌ Error during try-on: {str(e)}"}
        )
