import argparse
import pickle
import sys
from configparser import ConfigParser

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .logger import Logger
SHOW_LOG = True


class Predictor():

    def __init__(self, config: ConfigParser) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.config = config
        self.parser = argparse.ArgumentParser(description="Predictor")
        self.parser.add_argument("-m",
                                 "--model",
                                 type=str,
                                 help="Select model",
                                 required=False,
                                 default="RandomForestClassifier",
                                 nargs="?",
                                 choices=["RandomForestClassifier"])
        self.parser.add_argument("-t",
                                 "--tests",
                                 type=str,
                                 help="Select tests",
                                 required=False,
                                 default="smoke",
                                 const="smoke",
                                 nargs="?",
                                 choices=["smoke", "func"])

        args = self.parser.parse_args()
        try:
            model_path = self.config.get('models.fitted', args.model)
            with open(model_path, "rb") as model:
                self.model: BaseEstimator = pickle.load(model)
            scaler_path = self.config.get('models.fitted', 'StandardScaler')
            with open(scaler_path, "rb") as scaler:
                self.standard_scaler: BaseEstimator = pickle.load(scaler)
        except FileNotFoundError as e:
            self.log.error(e)
            sys.exit(1)

    @classmethod
    def from_pretrained(cls, config: ConfigParser):
        return cls(config)

    def predict(self, vector: list[list[float]]) -> np.array:
        vector = self.standard_scaler.transform(vector)
        return self.model.predict(vector)

    def test(self):
        data_split = self.config['data.splited']
        x_test_path = data_split['x_test']
        y_test_path = data_split['y_test']

        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).iloc[:, 0]

        predicts = self.model.predict(x_test)
        args = self.parser.parse_args()
        if args.tests == "smoke":
            try:
                score = self.model.score(x_test, y_test)
                self.log.info(f'{args.model} has {score} score')
            except Exception as e:
                self.log.error(e)
                sys.exit(1)
            self.log.info(f'{args.model} passed smoke tests')


if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini")
    predictor = Predictor(config)
    predictor.test()
