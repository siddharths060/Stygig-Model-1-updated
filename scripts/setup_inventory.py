import os
import pandas as pd
from PIL import Image

# Create directories
os.makedirs('data/inventory/images', exist_ok=True)

# Define dummy data
items = [
    {'id': 1, 'filename': 'red_dress.jpg', 'category': 'dress', 'tags': 'casual, red', 'color': (255, 0, 0)},
    {'id': 2, 'filename': 'blue_shirt.jpg', 'category': 'shirt', 'tags': 'formal, blue', 'color': (0, 0, 255)},
    {'id': 3, 'filename': 'green_jeans.jpg', 'category': 'jeans', 'tags': 'casual, green', 'color': (0, 255, 0)},
    {'id': 4, 'filename': 'yellow_skirt.jpg', 'category': 'skirt', 'tags': 'party, yellow', 'color': (255, 255, 0)},
    {'id': 5, 'filename': 'purple_blouse.jpg', 'category': 'blouse', 'tags': 'elegant, purple', 'color': (128, 0, 128)},
    {'id': 6, 'filename': 'orange_jacket.jpg', 'category': 'jacket', 'tags': 'winter, orange', 'color': (255, 165, 0)},
    {'id': 7, 'filename': 'pink_sweater.jpg', 'category': 'sweater', 'tags': 'cozy, pink', 'color': (255, 192, 203)},
    {'id': 8, 'filename': 'brown_pants.jpg', 'category': 'pants', 'tags': 'business, brown', 'color': (165, 42, 42)},
    {'id': 9, 'filename': 'gray_suit.jpg', 'category': 'suit', 'tags': 'formal, gray', 'color': (128, 128, 128)},
    {'id': 10, 'filename': 'black_tshirt.jpg', 'category': 'tshirt', 'tags': 'casual, black', 'color': (0, 0, 0)},
]

# Generate images
for item in items:
    img = Image.new('RGB', (224, 224), item['color'])
    img.save(f'data/inventory/images/{item["filename"]}')

# Create metadata.csv
df = pd.DataFrame(items)
df = df[['id', 'filename', 'category', 'tags']]
df.to_csv('data/metadata.csv', index=False)

print("Inventory setup complete.")