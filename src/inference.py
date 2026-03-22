import cv2
import numpy as np

from body_shape import detect_body_shape
from skin_tone import detect_skin_tone
from clip_model import get_image_embedding
from faiss_search import load_index, search_similar
from compatibility_model import OutfitCompatibilityModel, predict_compatibility
from sagemaker_model2 import call_model2
from color_recommender import recommend_colors


# -----------------------------
# Load heavy resources ONCE
# -----------------------------

FAISS_INDEX = load_index()

COMPATIBILITY_MODEL = OutfitCompatibilityModel()


# -----------------------------
# Utility
# -----------------------------

def load_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# -----------------------------
# Pipeline
# -----------------------------

def run_pipeline(image_path):

    image = load_image(image_path)

    # Body shape
    body_shape = detect_body_shape(image)

    # Skin tone
    skin_tone = detect_skin_tone(image)

    # Color recommendation
    recommended_colors = recommend_colors(skin_tone)

    # CLIP embedding
    embedding = get_image_embedding(image)

    # FAISS similarity search
    indices, distances = search_similar(FAISS_INDEX, embedding)

    # Compatibility scoring
    compatibility_scores = []

    for i in indices[0]:

        item_embedding = np.random.rand(1, 512)

        score = predict_compatibility(
            COMPATIBILITY_MODEL,
            embedding,
            item_embedding
        )

        compatibility_scores.append(score)

    # Call SageMaker model
    with open(image_path, "rb") as f:

        image_bytes = f.read()

    model2_result = call_model2(image_bytes)

    result = {

        "body_shape": body_shape,
        "skin_tone": skin_tone,
        "recommended_colors": recommended_colors,
        "similar_items": indices.tolist(),
        "compatibility_scores": compatibility_scores,
        "model2_output": model2_result

    }

    return result


# -----------------------------
# Local test
# -----------------------------

if __name__ == "__main__":

    test_image = "test.jpg"

    output = run_pipeline(test_image)

    print("Final Recommendation Output:")
    print(output)