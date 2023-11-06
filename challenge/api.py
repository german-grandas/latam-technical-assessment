import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = fastapi.FastAPI()

model = None

class InputBaseModel(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class InputData(BaseModel):
    features: List[InputBaseModel]


@app.on_event("startup")
async def startup_event():
    global model
    model = joblib.load('deplay_log_reg_model.pkl')

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: InputData) -> dict:
    if model is None:
        raise HTTPException(status_code=status.HTTP_503, detail="Model is not loaded yet")

    features = data.features
    prediction = model.predict(features)

    return {'predict': predictions.tolist()}