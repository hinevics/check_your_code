import os
from dotenv import load_dotenv


load_dotenv()

TG_TOKEN = os.getenv('TG_TOKEN')
PATH_DATA = os.getenv('PATH_DATA')
PATH_MODELS = os.getenv('PATH_MODELS')
LOG_PATH = os.getenv('LOG_PATH')
LOG_ERROR_PATH = os.getenv('LOG_ERROR_PATH')
PATH_TEMP = os.getenv('PATH_TEMP')

NAME_MODEL = r'model_v1.bin'
