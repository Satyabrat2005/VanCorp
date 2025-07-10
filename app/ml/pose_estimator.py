import mediapipe as mp
import cv2
import numpy as np 

mp_pose = mp.solutions.pose # type: ignore

def estimate_pose(user_image: np.ndarray) -> dict:
    """
    Estimate keypoints using MediaPipe Pose.
    Returns a dictionary of body keypoints.
    """
    pose = mp_pose.Pose(static_image_mode = True)
    results = pose.process(cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        raise ValueError("No Pose landmark detected")
    
    h,w,_ = user_image.shape
    landmarks = results.pose_landmarks.landmark

    def get_point(landmark):
        return (int(landmark.x * w), int(landmark.y * h))
    
    return {
        "left_shoulder": get_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]),
        "right_shoulder": get_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
        "left_hip": get_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP]),
        "right_hip": get_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP]),
    }

