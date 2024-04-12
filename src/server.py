import warnings
from configparser import ConfigParser
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .predict import Predictor

app = FastAPI()

config = ConfigParser()
config.read('config.ini')
predictor = Predictor.from_pretrained(config)


class PredictModel(BaseModel):
    x: list[list[float]] = Field(..., example=[
                                 [3.6216, 8.6661, -2.8073, -0.44699]])
    y: list[float] | None = None


@app.post("/predict")
def predict(items: PredictModel):
    x = items.x
    y_true = items.y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        y_pred = predictor.predict(x).tolist()
    return {'x': x, 'y_pred': y_pred, 'y_true': y_true}


if __name__ == "__main__":
    adress = config.get('server', 'adress')
    port = config.getint('server', 'port')
    uvicorn.run('src.server:app', host=adress, port=port, reload=True)
