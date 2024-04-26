import json
import warnings
from configparser import ConfigParser
from typing import Any

from httpx import request
import uvicorn
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from .database import Database

from .predict import Predictor
from .schemas import PredictionModel, PredictedModel

app = FastAPI()

config = ConfigParser()
config.read('config.ini')
predictor = Predictor.from_pretrained(config)

db = Database()


@app.post("/predict")
def predict(items: PredictionModel, session: Session = Depends(db.get_session)):
    x = items.x
    y_true = items.y_true
    if y_true is None:
        y_true = [None] * len(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        y_pred = predictor.predict(x).tolist()
    result = PredictedModel(x=x, y_pred=y_pred, y_true=y_true)
    db.create_record(session, result)
    return {"x": result.x, "y_true": result.y_true, "y_pred": result.y_pred}


@app.get('/predictions', response_model=list[PredictedModel])
def get_predictions(session: Session = Depends(db.get_session)):
    records = db.get_predictions(session)
    for record in records:
        record.x = json.loads(record.x)
    return records

@app.get('/prediction/last', response_model=PredictedModel | dict)
def get_last_prediction(session: Session = Depends(db.get_session)):
    record = db.get_last_prediction(session)
    if record is None:
        return {"Error": "No one record in DB"}
    record.x = json.loads(record.x)
    return record


if __name__ == "__main__":
    adress = config.get('server', 'adress')
    port = config.getint('server', 'port')
    uvicorn.run('src.server:app', host=adress, port=port, reload=True)
