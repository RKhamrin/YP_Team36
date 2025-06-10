import os
import joblib
import numpy as np

MODELS_PATH = os.path.join(os.path.dirname(__file__), '../model_trainer/data')

model = joblib.load(os.path.join(MODELS_PATH, 'baseline_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_PATH, 'baseline_scaler.pkl'))
ohe = joblib.load(os.path.join(MODELS_PATH, 'baseline_ohe.pkl'))

def make_prediction(team_name: str):
    features = np.zeros((1, 10))
    features_scaled = scaler.transform(features)
    features_ohe = ohe.transform(features_scaled)
    prediction = model.predict(features_ohe)
    return prediction[0] 