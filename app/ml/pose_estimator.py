import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, Tuple

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose  # type: ignore
POSE_LANDMARKS = mp_pose.PoseLandmark


def classify_pose(landmarks, image_shape: Tuple[int, int, int]) -> str:
    """
    Classifies user's pose based on landmark positions.

    Args:
        landmarks: MediaPipe landmarks.
        image_shape: Shape of the input image.

    Returns:
        str: Pose class - 'standing', 'sitting', or 'turning'.
    """
    h, w, _ = image_shape

    def to_xy(landmark):
        return np.array([landmark.x * w, landmark.y * h])

    left_hip = to_xy(landmarks[POSE_LANDMARKS.LEFT_HIP])
    right_hip = to_xy(landmarks[POSE_LANDMARKS.RIGHT_HIP])
    left_knee = to_xy(landmarks[POSE_LANDMARKS.LEFT_KNEE])
    right_knee = to_xy(landmarks[POSE_LANDMARKS.RIGHT_KNEE])
    left_shoulder = to_xy(landmarks[POSE_LANDMARKS.LEFT_SHOULDER])
    right_shoulder = to_xy(landmarks[POSE_LANDMARKS.RIGHT_SHOULDER])

    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
    avg_knee_y = (left_knee[1] + right_knee[1]) / 2
    vertical_ratio = avg_knee_y / (avg_hip_y + 1e-6)

    shoulder_diff = abs(left_shoulder[0] - right_shoulder[0])
    hip_diff = abs(left_hip[0] - right_hip[0])
    turning_ratio = shoulder_diff / (hip_diff + 1e-6)

    if vertical_ratio >= 1.2:
        return "sitting"
    elif turning_ratio < 0.5:
        return "turning"
    else:
        return "standing"


def estimate_pose(user_image: np.ndarray, draw: bool = False) -> Dict[str, Tuple[int, int]]:
    """
    Estimates user keypoints and classifies pose mode using MediaPipe.

    Args:
        user_image (np.ndarray): Input user image.
        draw (bool): Draw landmarks on image if True.

    Returns:
        Dict[str, Tuple[int, int]]: Keypoints and pose_mode.
    """
    h, w, _ = user_image.shape

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        rgb_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        if not results.pose_landmarks:
            raise ValueError("❌ No pose landmarks detected.")

        landmarks = results.pose_landmarks.landmark
        keypoints = {}

        def get_coords(landmark):
            return (int(landmark.x * w), int(landmark.y * h))

        keypoints["left_shoulder"] = get_coords(landmarks[POSE_LANDMARKS.LEFT_SHOULDER])
        keypoints["right_shoulder"] = get_coords(landmarks[POSE_LANDMARKS.RIGHT_SHOULDER])
        keypoints["left_hip"] = get_coords(landmarks[POSE_LANDMARKS.LEFT_HIP])
        keypoints["right_hip"] = get_coords(landmarks[POSE_LANDMARKS.RIGHT_HIP])

        # Add classified pose mode
        pose_mode = classify_pose(landmarks, user_image.shape) # type: ignore
        keypoints["pose_mode"] = pose_mode

        # Optional visualization
        if draw:
            mp_drawing = mp.solutions.drawing_utils  # type: ignore
            annotated = user_image.copy()
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Pose Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return keypoints
