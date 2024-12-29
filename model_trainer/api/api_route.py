from typing import Union, Dict, List

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from http import HTTPStatus

from pydantic import BaseModel
import pickle 

import json

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from api.utils import GetTrain, GetPrediction
from dataclasses import dataclass


router = APIRouter(prefix='/api/models')

models = {}
models_load = {}

with open("data/baseline_model.pkl", 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open("data/baseline_ohe.pkl", 'rb') as ohe_file:
    loaded_ohe = pickle.load(ohe_file)
with open("data/baseline_scaler.pkl", 'rb') as model_file:
    loaded_scaler = pickle.load(model_file)
models['baseline_model'] = [loaded_model, loaded_ohe, loaded_scaler, {}]

class Item(BaseModel):
    Attendance: int
    Performance3: float
    Standard5: float
    Poss: float
    Standard4: float
    Standard3: float
    offside: int
    crosses: int
    fouls_drw: int
    fouls_com: int
    Int: int
    Tackles1: int
    Standard1: int
    Standard2: int
    Performance2: int
    Performance: int
    Performance4: int
    OwnGoals: int
    Ast: int
    PenaltyKicks1: int
    PenaltyKicks2: int
    PenaltyKicks3: int
    PenaltyKicks: int
    Standard9: int
    Tkl_Int: int
    sec_yel: int
    red: int
    yellow: int
    Standard8: int
    GA: int
    GF: int
    date: str
    team: str
    venue: str
    opponent: str

class Items(BaseModel):
    objects: List[Item]

class ModelConfig(BaseModel):
    id: str
    ml_model_type: str
    hyperparameters: Dict

class FitResponse(BaseModel):
    message: str

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
    """Функция демонстрации 
    
    """
    data = pd.read_csv('data/data_sample.csv')
    data.columns = list(map(lambda x: x.replace('.', ''), data.columns))
    sample_data = data.sample(20)

    stream = io.StringIO()
    sample_data.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=sample_data.csv"
    return response

@router.post("/fit", response_model=FitResponse, status_code=HTTPStatus.CREATED)
async def fit(jsonfile: str, data: UploadFile):
    """json: {'model_id': str, 'hyperparameters': {}}

    """
    data = pd.read_csv(data.file, index_col=0).reset_index().drop(columns=['index'])
    data.sort_values(by = 'date', ignore_index = 'True', inplace = True)
    config = json.loads(jsonfile)
    model_id, hyperparameters = config['model_id'], config['hyperparameters']

    feat_data = GetTrain(data)
    print(len(data.columns))
    x = pd.DataFrame(feat_data, columns = data.drop(['team','date', 'opponent', 'venue', 'result'], axis = 1).columns)
    x = pd.concat([x, data['venue']], axis = 1)

    for_enc = x[['venue']]
    enc = OneHotEncoder(handle_unknown='ignore',drop = 'first', sparse_output = False)
    encd = enc.fit_transform(for_enc)
    one_hot_df = pd.DataFrame(encd, columns=enc.get_feature_names_out(['venue']))

    x = pd.concat([x.drop(['venue'], axis = 1),one_hot_df], axis = 1)
    print(data.columns)
    y = list(map((lambda i: 1 if i == 'W' else 0), data['result']))
    
    normalizer = StandardScaler()
    scaler = normalizer.fit(x)
    x = scaler.transform(x)

    model = LogisticRegression(**hyperparameters)
    model.fit(x, y)

    global models
    models[model_id] = hyperparameters

    with open(f"data/{model_id}_model.pkl", 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(f"data/{model_id}_ohe.pkl", 'wb') as ohe_file:
        pickle.dump(enc, ohe_file)
    with open(f"data/{model_id}_scaler.pkl", 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    return {'message': f'model {model_id} is trained and saved'}

@router.post("/predict")
async def predict(jsonfile: str, data: UploadFile) -> StreamingResponse:
    """json: {'model_id': str}

    """
    data = pd.read_csv(data.file, index_col=0).reset_index().drop(columns=['index'])
    data.sort_values(by = 'date', ignore_index = 'True', inplace = True)
    config = json.loads(jsonfile)
    model_id = config['model_id']
    print(loaded_model)
    with open(f"data/{model_id}_model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)
    with open(f"data/{model_id}_ohe.pkl", 'rb') as ohe_file:
        enc = pickle.load(ohe_file)
    with open(f"data/{model_id}_scaler.pkl", 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    stream = io.StringIO()
    data['preds'] = GetPrediction(data, enc, scaler, model)
    data.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

@router.get("/show_models", response_model=ShowModelsResponse)
async def show_models():
    """
    
    """

    models_info = {}
    for model_id in models:
        models_info[model_id] = {'hyperparameters': models[model_id]}

    return ShowModelsResponse(message = models_info)

@router.post("/set_model", response_model=SetModelResponse)
async def set_model(request: SetModelRequest):
    """

    """

    model_id = request.model_dump()['id']
    global models_load
    models_load[model_id] = models[model_id]

    return SetModelResponse(message = f'Model {model_id} is loaded')
