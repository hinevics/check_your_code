import os
from dotenv import load_dotenv


load_dotenv()

TG_TOKEN = os.getenv('TG_TOKEN')
PATH_DATA = os.getenv('PATH_DATA')
PATH_MODELS = os.getenv('PATH_MODELS')
