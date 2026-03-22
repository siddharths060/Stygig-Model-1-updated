
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose


def detect_body_shape(image):
    """
    Detect body shape based on shoulder and hip ratio
    """

    pose = mp_pose.Pose(static_image_mode=True)

    results = pose.process(image)

    if not results.pose_landmarks:
        return "unknown"

    landmarks = results.pose_landmarks.landmark

    # Shoulder width
    left_shoulder = landmarks[11].x
    right_shoulder = landmarks[12].x

    # Hip width
    left_hip = landmarks[23].x
    right_hip = landmarks[24].x

    shoulder_width = abs(left_shoulder - right_shoulder)
    hip_width = abs(left_hip - right_hip)

    ratio = shoulder_width / (hip_width + 1e-6)

    if ratio > 1.25:
        return "inverted_triangle"

    elif ratio < 0.85:
        return "triangle"

    else:
        return "rectangle"
