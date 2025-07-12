import cv2
from app.ml.tryon_pipeline import run_virtual_tryon

def test_run_virtual_tryon_output():
    user_path = "app/static/uploads/users/sample_user.jpg"
    cloth_path = "app/static/uploads/cloths/sample_cloth.jpg"

    result = run_virtual_tryon(user_path, cloth_path)
    assert isinstance(result, (cv2.typing.MatLike, type(None))) # type: ignore
    assert result is not None
    assert result.shape[0] > 0 and result.shape[1] > 0