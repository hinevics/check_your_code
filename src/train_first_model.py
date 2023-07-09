import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
import glob

from gensim.models.doc2vec import Doc2Vec

from processing import code_tokenize
import logger
from myconfig import PATH_TEMP
from texts.example_code import test_doc_bad, test_doc_best


def load_data(path: pathlib.PosixPath) -> pd.DataFrame:
    list_files = glob.glob(path, '*.csv')
    data = pd.concat([pd.read_csv(name) for name in list_files])
    return data


def main():
    data = load_data(PATH_TEMP)

    logger.logger.info('START -- crate test data --')
    tokens_test_doc_best = code_tokenize(test_doc_best)
    tokens_test_doc_bad = code_tokenize(test_doc_bad)
    logger.logger.info('COMPLETED -- crate test data --')

    logger.logger.info('START -- learn model --')
    model = Doc2Vec(
        vector_size=200,
        window=50,
        min_count=2,
        epochs=10,
        alpha=0.025,
        min_alpha=0.025)
    if 'tagged_docs' not in data.columns:
        raise KeyError('tagged_docs not in columns')
    model.build_vocab(list(data.tagged_docs.values))

    message_info = "train model"
    logger.logger.info(f'START: {message_info}')

    for epoch in range(10):
        if epoch % 2 == 0:
            logger.logger.debug(f'now training epoch {epoch}\n')
        model.train(
            code_data.tagged_docs.values,
            total_examples=model.corpus_count, epochs=model.epochs)
        valid_tokens = code_data.iloc[0].tokens
        get_metrics_unobservable(
            tokens_doc_best=tokens_test_doc_best,
            tokens_doc_bad=tokens_test_doc_bad,
            tokens_valid=valid_tokens,
            model=model
        )
        model.alpha -= 0.002
        model.min_alpha = model.alpha


if __name__ == '__main__':
    main()
