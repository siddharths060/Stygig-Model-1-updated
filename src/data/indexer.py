import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os

class FashionIndexer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embeddings = []
        self.filenames = []

    def index_inventory(self, image_folder):
        for file in os.listdir(image_folder):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_folder, file)
                image = Image.open(image_path)
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    embedding = self.model.get_image_features(**inputs)
                # Normalize the embedding
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                self.embeddings.append(embedding.squeeze().numpy())
                self.filenames.append(file)

    def save_data(self, output_path):
        np.save(output_path, np.array(self.embeddings))
        print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    indexer = FashionIndexer()
    indexer.index_inventory("data/inventory/images")
    indexer.save_data("data/embeddings.npy")
    print("Indexing complete.")