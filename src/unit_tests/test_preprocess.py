import unittest
from configparser import ConfigParser

from ..preprocess import DataPreprocessor


class TestPreprocess(unittest.TestCase):

    def setUp(self) -> None:
        self.config = ConfigParser()
        self.config.read('config.ini')
        self.preprocess = DataPreprocessor(self.config)

    def test_split_data(self):
        self.assertEqual(len(self.preprocess.split_data()), 4)

    def test_standard_data(self):
        self.preprocess.split_data()
        X_train, X_test, y_train, y_test = self.preprocess.standard_data()
        mean = X_train.mean().to_list()
        std = X_train.std().to_list()
        for elem in mean:
            self.assertAlmostEqual(elem, 0, places=7)
        for elem in std:
            self.assertAlmostEqual(elem, 1, places=3)


if __name__ == "__main__":
    """python -m src.unit_tests.test_preprocess"""
    unittest.main()
