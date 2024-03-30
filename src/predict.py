import argparse
import pickle
import sys
from configparser import ConfigParser
from datetime import datetime

import numpy as np
from sklearn.base import BaseEstimator

# from logger import Logger

SHOW_LOG = True


class Predictor():

    def __init__(self) -> None:
        # logger = Logger(SHOW_LOG)
        self.config = ConfigParser()
        # self.log = logger.get_logger(__name__)
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

    def predict(self, vector) -> bool:
        vector = np.array(vector)
        args = self.parser.parse_args()
        try:
            model_path = self.config.get('models.fitted', args.model)
            with open(model_path, "rb") as model:
                classifier: BaseEstimator = pickle.load(model)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

        return classifier.predict(vector)


if __name__ == "__main__":
    predictor = Predictor()
    print(predictor.predict([[1, 2, 3, 4]]))
