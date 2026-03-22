import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_embedding(image):
    """
    Generate embedding vector for an image
    """

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    return embedding.cpu().numpy()