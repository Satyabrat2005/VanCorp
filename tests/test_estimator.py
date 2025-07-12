import cv2
from app.ml.pose_estimator import estimate_pose

def test_estimate_pose_keypoints():
    image = cv2.imread("app/static/uploads/users/sample_user.jpg")
    assert image is not None, "Sample image not found"

    keypoints = estimate_pose(image)
    assert "left_shoulder" in keypoints
    assert "right_shoulder" in keypoints
    assert all(isinstance(pt, tuple) for pt in keypoints.values())