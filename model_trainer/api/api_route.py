from typing import Union, Dict, List
import pickle
import json

import io
from io import StringIO
from http import HTTPStatus
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from catboost import CatBoostClassifier

from api.utils import GetTrain, GetPrediction, getStats


router = APIRouter(prefix='/api/models')

models = {}
models_load = {}

data = pd.read_csv('model_trainer/data_final_2.csv', index_col=0)

with open("model_trainer/data/baseline_model.pkl", 'rb') as baseline_model:
    baseline_model = pickle.load(baseline_model)
with open("model_trainer/data/baseline_ohe.pkl", 'rb') as baseline_ohe:
    baseline_ohe = pickle.load(baseline_ohe)
with open("model_trainer/data/baseline_scaler.pkl", 'rb') as baseline_scaler:
    baseline_scaler = pickle.load(baseline_scaler)

with open("model_trainer/data/rf_model.pkl", 'rb') as baseline_model:
    baseline_model = pickle.load(baseline_model)
with open("model_trainer/data/rf_ohe.pkl", 'rb') as baseline_ohe:
    baseline_ohe = pickle.load(baseline_ohe)
with open("model_trainer/data/rf_scaler.pkl", 'rb') as baseline_scaler:
    baseline_scaler = pickle.load(baseline_scaler)

models['baseline_model'] = {}
models['rf_model'] = {'max_depth':10}


class FitResponse(BaseModel):
    message: str

class PredictRequest(BaseModel):
    model_id: str
    home_team: str
    arrive_team: str


class PredictResponse(BaseModel):
    score: str


class ShowModelsResponse(BaseModel):
    message: Dict


class SetModelRequest(BaseModel):
    id: str


class SetModelResponse(BaseModel):
    message: str


class ValidationError(HTTPException):
    loc: List[Union[str, int]]
    msg: str
    type: str


class ApiResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None


# API endpoints
@router.get("/show_example")
async def show_example() -> StreamingResponse:
    """Функция получения примера (сэмпла) данных для дообучения модели
    returns:
        data: csv
    """
    sample_data = data.sample(20).copy()

    stream = io.StringIO()
    sample_data.to_csv(stream)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=sample_data.csv"
    return response


@router.post("/fit", response_model=FitResponse, status_code=HTTPStatus.CREATED)
async def fit(jsonfile: str, extra_data: UploadFile = File(...)):
    """Функция обучения модели с возможностью дообучения текущей 
    params:
        jsonfile: {'model_id': str, 'hyperparameters': Dict, 'model_type': str}
        model_type: linear, boosting, bagging, random_forest
        data: csv

    returns:
        {"message": "Model model_id is trained and saved"}
    """
    content = StringIO(extra_data.file.read().decode("utf-8"))
    extra_data = pd.read_csv(content, index_col=0)
    if extra_data.shape[0] != 0:
        extra_data = extra_data.sort_values(by='date', ignore_index='True')
        
        config = json.loads(jsonfile)
        model_id, hyperparameters, model_type = config['model_id'], config['hyperparameters'], config['model_type']
        data_full = pd.concat([data, extra_data], axis=0).copy().drop_duplicates()
        feat_data = GetTrain(data_full)

    x = pd.DataFrame(feat_data, columns=data.drop(['team', 'date', 'opponent', 'venue', 'result'], axis=1).columns)
    x = pd.concat([x, data['venue']], axis=1)

    enc = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    encd = enc.fit_transform(x[['venue']])
    one_hot_df = pd.DataFrame(encd, columns=enc.get_feature_names_out(['venue']))

    x = pd.concat([x.drop(['venue'], axis=1), one_hot_df], axis=1)
    y = list(map((lambda i: 1 if i == 'W' else 0), data['result']))

    normalizer = StandardScaler()
    scaler = normalizer.fit(x)
    x = scaler.transform(x)
    
    if model_type == 'linear':
        model = LogisticRegression(**hyperparameters)
    elif model_type == 'boosting':
        model = CatBoostClassifier(**hyperparameters)
    elif model_type == 'bagging':
        model = BaggingClassifier(**hyperparameters)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**hyperparameters)

    model.fit(x, y)
    models[model_id] = hyperparameters

    with open(f"model_trainer/data/{model_id}_model.pkl", 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(f"model_trainer/data/{model_id}_ohe.pkl", 'wb') as ohe_file:
        pickle.dump(enc, ohe_file)
    with open(f"model_trainer/data/{model_id}_scaler.pkl", 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return {'message': f'model {model_id} is trained and saved'}


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Функция получения предсказаний загруженной модели
    params:
        model_id: str
        home_team: str
        opponent_team: str

    returns:
        {"score": 1 в случае победы или ничьей домашней команды, 0 в противном случае}
    """

    data_local = data.sort_values(by='date', ignore_index='True').copy()
    config = request.model_dump()
    model_id, home_team, opponent_team = config['model_id'], config['home_team'], config['opponent_team']

    with open(f"model_trainer/data/{model_id}_model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)
    with open(f"model_trainer/data/{model_id}_ohe.pkl", 'rb') as ohe_file:
        enc = pickle.load(ohe_file)
    with open(f"model_trainer/data/{model_id}_scaler.pkl", 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # data_local[(data_local['team'] == home_team)&(data_local['opponent'] == opponent_team)]
    home_stats = getStats(home_team, '2025-01-01', data_local)
    opponent_stats = getStats(opponent_team, '2025-01-01', data_local)
    features = [[a - b for a, b in zip(home_stats, opponent_stats)]]

    x = pd.DataFrame(features, columns=data.drop(['team', 'date', 'opponent', 'venue', 'result'], axis=1).columns)
    x['venue'] = 'Home'

    for_enc = x[['venue']]
    encd = enc.transform(for_enc)
    one_hot_df = pd.DataFrame(encd, columns=enc.get_feature_names_out(['venue']))

    x = pd.concat([x.drop(['venue'], axis=1), one_hot_df], axis=1)
    x = scaler.transform(x)
    preds = model.predict(x)
    return preds


@router.get("/show_models", response_model=ShowModelsResponse)
async def show_models():
    """Функция получения списка моделей

    returns:
        models_info: Dict
    """

    models_info = {}
    for model_id, hyps in models.items():
        models_info[model_id] = {'hyperparameters': hyps}

    return ShowModelsResponse(message=models_info)


@router.post("/set_model", response_model=SetModelResponse)
async def set_model(request: SetModelRequest):
    """Функция загрузки модели
    params:
        id: str

    returns:
        {"message": "Model id is loaded"}

    """
    model_id = request.model_dump()['id']
    models_load[model_id] = models[model_id]

    return SetModelResponse(message=f'Model {model_id} is loaded')
