import os
import sys
import unittest
from configparser import ConfigParser

import numpy as np
import pandas as pd

from ..train import Trainer

from sklearn.ensemble import RandomForestClassifier


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.config = ConfigParser()
        self.config.read('config.ini')
        self.model = RandomForestClassifier()
        self.trainer = Trainer(self.config, self.model)

    def test_train(self):
        x_train = pd.read_csv(self.config.get('data.splited', 'x_train'))
        y_train = pd.read_csv(self.config.get('data.splited', 'y_train'))
        trained_model = self.trainer.train(x_train, y_train)
        self.assertIs(self.model, trained_model)


if __name__ == "__main__":
    """python -m src.unit_tests.test_train"""
    unittest.main()