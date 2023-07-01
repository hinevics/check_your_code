# loading data

import pandas as pd
import pathlib

import logger


def load_data(path: pathlib.PosixPath) -> pd.DataFrame:
    # Функция для простой подгрудки документа
    if path.exists():
        return pd.read_csv(path)
    logger.error_logger('File not Exists!')
    raise FileExistsError('Файла нет!')
