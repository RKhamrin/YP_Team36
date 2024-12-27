from typing import Union, Dict, List

from fastapi import APIRouter, HTTPException
from http import HTTPStatus

from pydantic import BaseModel
import pickle 

import time

models = {}
models['models'] = []
models_load = {}

router = APIRouter(prefix='/api/v1/models')

class LoadRequest(BaseModel):
    id: str

class LoadResponse(BaseModel):
    message: str

class ValidationError(HTTPException):
    loc: List[Union[str, int]]
    msg: str
    type: str

# class HTTPException(BaseModel):
#     detail: ValidationError

class ApiResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None

# API endpoints
@router.post("/load", response_model=List[LoadResponse])
async def load(request: LoadRequest):
    """Function of load model
    parameters:
        id: str

    returns:
        message: str
    """
    model_id = request.model_dump()['id']
    if model_id not in [i['id'] for i in models['models']]:
        raise HTTPException(status_code=422, detail=f"Model with id {model_id} is not found")
    with open(f'model_{model_id}.pkl', 'rb') as file: 
        model = pickle.load(file)
    global models_load
    models_load[model_id] = model
    return [LoadResponse(message = f"Model '{model_id}' loaded")]
