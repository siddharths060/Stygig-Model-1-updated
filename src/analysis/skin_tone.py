import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional

class SkinToneAnalyzer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    def get_skin_tone(self, image_path: str) -> str:
        image = cv2.imread(image_path)
        if image is None:
            return "Unknown"
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            points = []
            for lm in face_landmarks.landmark:
                points.append((int(lm.x * w), int(lm.y * h)))
            points = np.array(points)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)
            masked_pixels = image[mask == 255]
            if len(masked_pixels) == 0:
                return "Unknown"
            lab = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
            kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
            kmeans.fit(lab.reshape(-1, 3))
            dominant = kmeans.cluster_centers_[0]
            L = dominant[0]
            if L > 60:
                return "Light"
            else:
                return "Dark"
        return "Unknown"