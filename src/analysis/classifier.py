import xgboost as xgb
import numpy as np
from typing import Dict

class BodyShapeClassifier:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model('models/body_shape_xgb.json')

    def predict(self, landmarks_dict: Dict[str, float]) -> str:
        X = np.array([[landmarks_dict['shoulder_width'], landmarks_dict['hip_width'], landmarks_dict['waist_width']]])
        pred = self.model.predict(X)
        return pred[0]