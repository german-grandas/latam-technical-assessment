import os
from itertools import chain

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from .model import DelayModel

app = FastAPI()

model = None


class InputBaseModel(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class InputData(BaseModel):
    flights: List[InputBaseModel]


@app.on_event("startup")
async def startup_event():
    global model
    current_dir = os.path.dirname(__file__)
    data_path = target_file_path = os.path.abspath(
        os.path.join(current_dir, "..", "data", "data.csv")
    )
    train_data = pd.read_csv(filepath_or_buffer=target_file_path)

    model = DelayModel()
    features, target = model.preprocess(data=train_data, target_column="delay")
    model.fit(features=features, target=target)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(data: InputData) -> dict:
    global model
    if model is None:
        current_dir = os.path.dirname(__file__)
        data_path = target_file_path = os.path.abspath(
            os.path.join(current_dir, "..", "data", "data.csv")
        )
        train_data = pd.read_csv(filepath_or_buffer=target_file_path)

        model = DelayModel()
        features, target = model.preprocess(data=train_data, target_column="delay")
        model.fit(features=features, target=target)

    features = data.model_dump()
    # Preprocessing
    features = features.get("flights")
    features = [[f"{key}_{value}" for key, value in item.items()] for item in features]

    model_features = set(model.features_cols)
    all_features = set(chain(*features))
    different_columns = all_features - model_features

    if len(different_columns) != 0:
        raise HTTPException(
            status_code=400, detail=f"Some unkwown columns: {different_columns}"
        )

    y = [
        [
            1 if model_expected_variable in feature else 0
            for model_expected_variable in model_features
        ]
        for feature in features
    ]
    predictions = model.predict(y)

    return {"predict": predictions}
