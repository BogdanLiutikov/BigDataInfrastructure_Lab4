import unittest
from configparser import ConfigParser

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..train import Trainer


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.config = ConfigParser()
        self.config.read('config.ini')
        self.model = RandomForestClassifier()
        self.trainer = Trainer(self.config, self.model)

    def test_train(self):
        x_train = pd.read_csv(self.config.get('data.splited', 'x_train'))
        y_train = pd.read_csv(self.config.get('data.splited', 'y_train')).iloc[:, 0]
        trained_model = self.trainer.train(x_train, y_train)
        self.assertIsNotNone(trained_model)
        self.assertTrue(self.trainer.fitted)

    def test_eval(self):
        x_train = pd.read_csv(self.config.get('data.splited', 'x_train'))
        y_train = pd.read_csv(self.config.get('data.splited', 'y_train')).iloc[:, 0]
        self.trainer.train(x_train, y_train)
        result = self.trainer.eval()
        self.assertTrue(result)


if __name__ == "__main__":
    """python -m src.unit_tests.test_train"""
    unittest.main()
