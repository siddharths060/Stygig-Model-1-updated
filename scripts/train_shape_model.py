import numpy as np
import xgboost as xgb
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Generate synthetic data
n = 2000
shoulder = np.random.uniform(0.1, 0.5, n)
hip = np.random.uniform(0.1, 0.5, n)
waist = np.random.uniform(0.05, 0.4, n)

labels = []
for i in range(n):
    s = shoulder[i]
    h = hip[i]
    w = waist[i]
    if s > h * 1.05:
        labels.append("Inverted Triangle")
    elif h > s * 1.05:
        labels.append("Pear")
    elif abs(s - h) < 0.05 and w < h * 0.75:
        labels.append("Hourglass")
    else:
        labels.append("Rectangle")

# Prepare data for training
X = np.column_stack((shoulder, hip, waist))
y = labels

# Train the model
model = xgb.XGBClassifier(random_state=42)
model.fit(X, y)

# Save the model
model.save_model('models/body_shape_xgb.json')

print("Body shape model trained and saved to models/body_shape_xgb.json")