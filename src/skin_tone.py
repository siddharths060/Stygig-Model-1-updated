import cv2
import numpy as np
from sklearn.cluster import KMeans


def detect_skin_tone(image):
    """
    Detect skin tone using LAB color clustering
    """

    # Convert RGB image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Reshape image into pixels
    pixels = lab_image.reshape((-1, 3))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(pixels)

    # Get dominant color
    dominant_color = kmeans.cluster_centers_[0]

    lightness = dominant_color[0]

    # Classify tone
    if lightness > 170:
        return "fair"

    elif lightness > 140:
        return "medium"

    else:
        return "dark"