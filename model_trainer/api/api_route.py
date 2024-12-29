from typing import Union, Dict, List

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from http import HTTPStatus

from pydantic import BaseModel
import pickle 

# import time
import json

import pandas as pd
import numpy as np
# import collections
# import random

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from api.utils import GetTrain, GetPrediction

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
    # OppFormation: str
    # Formation: str
    # Captain: str
    # Referee: str
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
    # result: str
    GA: int
    GF: int
    # time: str
    # game: str
    date: str
    team: str
    # day: str
    # venue: str
    # opponent: str
    # season: int
    # GT: int
    # Gdiff: int

class Items(BaseModel):
    objects: List[Item]

# class ShowExampleResponse(BaseModel):
#     data_sample: pd.DataFrame

class ModelConfig(BaseModel):
    id: str
    ml_model_type: str
    hyperparameters: Dict

# class FitRequest(BaseModel):
#     X: pd.DataFrame
#     y: List
#     config: ModelConfig

class FitResponse(BaseModel):
    message: str

# class PredictRequest(BaseModel):
#     id: str
#     X: List[List]

# class PredictResponse(BaseModel):
#     predictions: List

class ModelsResponse(BaseModel):
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

class JsonFile(BaseModel):
    json_file: Dict

# API endpoints
@router.get("/show_example")
async def show_example() -> StreamingResponse:
    """
    
    """
    data = pd.read_csv('data/sample.csv', index_col=0)
    # data = data.drop([
    #     'Opp Formation', 'Formation', 'Captain', 'Referee','result','time','game', 
    #     'season', 'day', 'venue', 'opponent','GT', 'Gdiff'], axis = 1
    #                 )
    sample_data = data.sample(20)

    stream = io.StringIO()
    sample_data.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=sample_data.csv"
    return response


@router.post("/fit", response_model=FitResponse, status_code=HTTPStatus.CREATED)
async def fit(data: UploadFile, jsonfile: JsonFile):
    """json: {'model_id': str, 'hyperparameters': {}}

    """
    data = pd.read_csv(data.file, index_col=0)
    with open(jsonfile.file, 'r') as file:
        config = json.loads(file)

    model_id, hyperparameters = config['model_id'], config['hyperparameters']
    feat_data = GetTrain(data)
    x = pd.DataFrame(feat_data, columns = data.drop(['team','date'], axis = 1).columns)
    x = pd.concat([x, data['venue']], axis = 1)

    for_enc = x[['venue']]
    enc = OneHotEncoder(handle_unknown='ignore',drop = 'first', sparse_output = False)
    encd = enc.fit_transform(for_enc)
    one_hot_df = pd.DataFrame(encd, columns=enc.get_feature_names_out(['venue']))

    x = pd.concat([x.drop(['venue'], axis = 1),one_hot_df], axis = 1)
    y = list(map((lambda i: 1 if i == 'W' else 0), data['result']))
    
    normalizer = StandardScaler()
    scaler = normalizer.fit(x)
    x = scaler.transform(x)

    model = LogisticRegression(hyperparameters)
    model.fit(x, y)

    models[model_id] = [model, enc, scaler, hyperparameters]

    with open(f"data/{model_id}_model.pkl", 'wb') as model_file:
        pickle.dump(model)
    with open(f"data/{model_id}_ohe.pkl", 'wb') as ohe_file:
        pickle.dump(enc)
    with open(f"data/{model_id}_scaler.pkl", 'wb') as model_file:
        pickle.dump(scaler)
    return {'message': f'model {model_id} is trained and saved'}

@router.post("/predict")
async def predict(data: UploadFile, jsonfile: JsonFile) -> StreamingResponse:
    """json: {'model_id': str}

    """
    data = pd.read_csv(data.file, index_col=0)
    with open(jsonfile.file, 'r') as file:
        config = json.loads(file)
    loaded_model, enc, scaler = models_load[config['model_id']][:3]

    stream = io.StringIO()
    data['preds'] = GetPrediction(data, enc, scaler, loaded_model)
    data.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

@router.get("/models", response_model=ModelsResponse)
async def models():
    """
    
    """

    models_info = {}
    for model_id in models:
        models_info[model_id] = models[model_id][-1]

    return [ModelsResponse(message = models_info)]

@router.post("/set_model", response_model=SetModelResponse)
async def set_model(request: SetModelRequest):
    """

    """

    model_id = request.model_dump()['id']
    models_load[model_id] = models[model_id]

    return [SetModelResponse(message = f'Model {model_id} is loaded')]
