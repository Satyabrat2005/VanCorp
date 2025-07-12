from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_upload_image():
    with open("app/static/uploads/users/sample_user.jpg", "rb") as image:
        response = client.post("/upload/image", files={"image": image})
    assert response.status_code == 200
    assert "✅" in response.json()["message"]