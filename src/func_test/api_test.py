import json
import os

from fastapi.testclient import TestClient

from ..logger import Logger
from ..server import app

SHOW_LOG = True


class FunctionalApiTest():

    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.log = Logger(SHOW_LOG).get_logger(__name__)
        self.test_dir = './tests'

    def test_predict(self, test_name: str):
        """Tests the /predict endpoint with data from tests"""
        test_path = os.path.join(self.test_dir, test_name)
        with open(test_path, 'r') as f:
            data: dict = json.load(f)
        data["y_true"] = data.pop("y")

        expected = data.copy()
        expected["y_pred"] = expected["y_true"]

        response = self.client.post("/predict", json=data)
        assert response.status_code == 200
        predicted = response.json()
        assert predicted == expected, f"Expected: {expected}\nbut predicted: {predicted}"

        self.log.info(f'Functional {test_name} passed')

    def test_all(self):
        for test in os.listdir(self.test_dir):
            self.test_predict(test)


if __name__ == "__main__":
    test_client = TestClient(app)
    func_tester = FunctionalApiTest(test_client)
    func_tester.log.info('Functional tests running...')
    func_tester.test_all()
    func_tester.log.info('Functional tests passed')
