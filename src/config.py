import os
from dotenv import load_dotenv


load_dotenv()

TG_TOKEN = os.getenv('TG_TOKEN')
PATH_DATA = os.getenv('PATH_DATA')
