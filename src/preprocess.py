from configparser import ConfigParser
from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .logger import Logger

SHOW_LOG = True


class DataPreprocessor:
    def __init__(self, config: ConfigParser) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.config = config
        self.data: pd.DataFrame = pd.read_csv(
            config.get('data.preprocess', 'raw_path'))

    def split_data(self, test_size: float = 0.2, seed: int = 42):
        X, Y = self.data.drop(columns=['class']), self.data['class']
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=test_size, random_state=seed)
        self.splited_data: tuple[pd.DataFrame, ...] = (X_train, X_test,
                                                       y_train, y_test)
        return self.splited_data

    def standard_data(self, save_scaler: bool = True):
        X_train, X_test, y_train, y_test = self.splited_data
        self.standard_scaler = StandardScaler()
        column_names = X_train.columns
        X_train = pd.DataFrame(self.standard_scaler.fit_transform(X_train),
                               columns=column_names)
        if save_scaler:
            save_path = Path(self.config.get('models', 'save_dir')).joinpath(
                'StandardScaler.pkl')
            with open(save_path, 'wb') as file:
                dump(self.standard_scaler, file)
            scaler_name = save_path.stem
            if not self.config.has_section('models.fitted'):
                self.config.add_section('models.fitted')
            self.config.set('models.fitted', scaler_name, save_path.as_posix())
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)

        X_test = pd.DataFrame(self.standard_scaler.transform(X_test),
                              columns=column_names)
        self.splited_data = X_train, X_test, y_train, y_test
        return self.splited_data

    def save_data(self, path: str):
        X_train, X_test, y_train, y_test = self.splited_data
        save_dir = Path(path)
        X_train_path = save_dir.joinpath('X_train.csv')
        X_test_path = save_dir.joinpath('X_test.csv')
        y_train_path = save_dir.joinpath('y_train.csv')
        y_test_path = save_dir.joinpath('y_test.csv')
        X_train.to_csv(X_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)
        self.config['data.splited'] = {'X_train': X_train_path,
                                       'X_test': X_test_path,
                                       'y_train': y_train_path,
                                       'y_test': y_test_path}
        with open('config.ini', 'w') as c:
            self.config.write(c)


if __name__ == "__main__":
    config = ConfigParser()
    config.read('config.ini')
    data_preprocessor = DataPreprocessor(config)
    data_preprocessor.log.info('Preprocessing...')
    data_preprocessor.split_data(test_size=config.getfloat('data.preprocess', 'test_size'),
                                 seed=config.getint('data.preprocess', 'seed'))
    data_preprocessor.standard_data()
    data_preprocessor.save_data(config.get('data.preprocess', 'output_path'))
    data_preprocessor.log.info('Preprocessing finished')
