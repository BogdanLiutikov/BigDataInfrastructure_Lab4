from configparser import ConfigParser
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .predict import Predictor

app = FastAPI()


class PredictModel(BaseModel):
    x: list[list[float]] = Field(..., example=[
                                 [3.6216, 8.6661, -2.8073, -0.44699]])
    y: list[float] | None = None


predictor = Predictor()


@app.post("/predict")
def predict(items: PredictModel):
    x = items.x
    if items.y:
        y_true = items.y
    y_pred = predictor.predict(x).tolist()
    return {'x': x, 'y_pred': y_pred, 'y_true': y_true}


if __name__ == "__main__":
    config = ConfigParser()
    config.read('config.ini')
    adress = config.get('server', 'adress')
    port = config.getint('server', 'port')
    uvicorn.run('src.server:app', host=adress, port=port, reload=True)
