from typing import Union, Dict, List
import pickle
import json

import io
from http import HTTPStatus
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from api.utils import GetTrain, GetPrediction


router = APIRouter(prefix='/api/models')

models = {}
models_load = {}

with open("data/baseline_model.pkl", 'rb') as baseline_model:
    baseline_model = pickle.load(baseline_model)
with open("data/baseline_ohe.pkl", 'rb') as baseline_ohe:
    baseline_ohe = pickle.load(baseline_ohe)
with open("data/baseline_scaler.pkl", 'rb') as baseline_scaler:
    baseline_scaler = pickle.load(baseline_scaler)
models['baseline_model'] = {}


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
    """Функция получения примера (сэмпла) данных
    returns:
        data: csv
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
async def fit(jsonfile: str, data: UploadFile) -> StreamingResponse:
    """Функция обучения модели
    params:
        jsonfile: {'model_id': str, 'hyperparameters': Dict}
        data: csv

    returns:
        {"message": "Model model_id is trained and saved"}
    """
    data = pd.read_csv(data.file, index_col=0).reset_index().drop(columns=['index'])
    data.sort_values(by='date', ignore_index='True', inplace=True)
    config = json.loads(jsonfile)
    model_id, hyperparameters = config['model_id'], config['hyperparameters']

    feat_data = GetTrain(data)
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

    model = LogisticRegression(**hyperparameters)
    model.fit(x, y)

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
    """Функция получения предсказаний загруженной модели
    params:
        jsonfile: {'model_id': str}
        data: csv

    returns:
        data: csv (со столбцом preds -- предсказания)
    """
    data = pd.read_csv(data.file, index_col=0).reset_index().drop(columns=['index'])
    data.sort_values(by='date', ignore_index='True', inplace=True)
    model_id = json.loads(jsonfile)['model_id']
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
