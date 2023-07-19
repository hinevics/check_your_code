from gensim.models import Doc2Vec
import pathlib
from sklearn.metrics.pairwise import cosine_similarity

from processing import code_tokenize
from myconfig import PATH_MODELS, NAME_MODEL

path_model = pathlib.Path(PATH_MODELS).joinpath(NAME_MODEL)


def load_model(path: pathlib.PosixPath):
    model = Doc2Vec.load(str(path))
    print(type(model))
    return model


def check(code1: str, code2: str) -> int:
    code1_tokens = code_tokenize(code1)
    code2_tokens = code_tokenize(code2)
    model = load_model(path_model)
    print(code1_tokens, code2_tokens)
    code1_vector = model.infer_vector(code1_tokens)
    code2_vector = model.infer_vector(code2_tokens)
    similarity = cosine_similarity([code1_vector], [code2_vector])[0][0]
    return similarity
