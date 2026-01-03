import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional

class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def get_landmarks(self, image_path: str) -> Optional[Dict[str, float]]:
        image = cv2.imread(image_path)
        if image is None:
            return None
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            hip_width = abs(left_hip.x - right_hip.x)
            waist_width = 0.85 * hip_width  # heuristic
            return {
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'waist_width': waist_width
            }
        return None