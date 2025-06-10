import os
import joblib
import numpy as np
import pandas as pd

MODELS_PATH = os.path.join(os.path.dirname(__file__), '../model_trainer/data')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../model_trainer/data/data_final_2.csv')

model = joblib.load(os.path.join(MODELS_PATH, 'baseline_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_PATH, 'baseline_scaler.pkl'))
ohe = joblib.load(os.path.join(MODELS_PATH, 'baseline_ohe.pkl'))

def make_prediction(team_name: str):
    df = pd.read_csv(DATA_PATH)
    row = df[df['team'].str.lower() == team_name.lower()]
    if row.empty:
        return 'Нет данных для предсказания'
    features = row.drop(['team'], axis=1).values
    features_scaled = scaler.transform(features)
    features_ohe = ohe.transform(features_scaled)
    prediction = model.predict(features_ohe)
    return prediction[0] 