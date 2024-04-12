import argparse
import pickle
import sys
from configparser import ConfigParser

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Predictor():

    def __init__(self) -> None:
        self.config = ConfigParser()
        self.config.read("config.ini")
        self.parser = argparse.ArgumentParser(description="Predictor")
        self.parser.add_argument("-m",
                                 "--model",
                                 type=str,
                                 help="Select model",
                                 required=True,
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
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

    def predict(self, vector: list[list[float]]) -> np.array:
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
                print(f'{args.model} has {score} score')
            except Exception as e:
                print(e)
                sys.exit(1)
            print(f'{args.model} passed smoke tests')


if __name__ == "__main__":
    predictor = Predictor()
    predictor.test()
