import mediapipe as mp

mp_pose = mp.solutions.pose

def detect_body_shape(image):

    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(image)

    if not results.pose_landmarks:
        return "unknown"

    lm = results.pose_landmarks.landmark

    shoulder = abs(lm[11].x - lm[12].x)
    hip = abs(lm[23].x - lm[24].x)

    ratio = shoulder/(hip+1e-6)

    if ratio > 1.25:
        return "inverted_triangle"

    if ratio < 0.85:
        return "triangle"

    return "rectangle"