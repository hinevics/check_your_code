import pandas as pd
import pathlib
import glob

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec

from processing import code_tokenize
import logger
from myconfig import PATH_TEMP, PATH_MODELS
from texts.example_code import test_doc_bad, test_doc_best


def load_data_chunk(path: pathlib.PosixPath) -> pd.DataFrame:
    path = f'{path}/*.csv'
    list_files = glob.glob(path)
    data = pd.concat([pd.read_csv(name) for name in list_files])
    return data


def get_metrics_unobservable(
        tokens_doc_best,
        tokens_doc_bad,
        tokens_valid,
        model):
    # эта функция используется если мы наблюдаем за сторонними документами
    # в обучении не участвуют

    vector_best = model.infer_vector(tokens_doc_best)
    vector_bad = model.infer_vector(tokens_doc_bad)
    vector_valid = model.infer_vector(tokens_valid)
    unseen_similarity_best = cosine_similarity([vector_best], [vector_valid])[0][0]
    unseen_similarity_bad = cosine_similarity([vector_bad], [vector_valid])[0][0]
    logger.logger.debug('unseen_similarity best:', unseen_similarity_best)
    logger.logger.debug('unseen_similarity bad:', unseen_similarity_bad)


def main():
    logger.logger.info('START: load_data_chunk')
    data = load_data_chunk(PATH_TEMP)
    logger.logger.info('COMPLETED: load_data_chunk')
    logger.logger.info('START: crate test data')
    tokens_test_doc_best = code_tokenize(test_doc_best)
    tokens_test_doc_bad = code_tokenize(test_doc_bad)
    logger.logger.info('COMPLETED: crate test data')

    logger.logger.info('START: train model')
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
    for epoch in range(10):
        if epoch % 2 == 0:
            logger.logger.debug(f'Now training epoch {epoch}\n')
        model.train(
            data.tagged_docs.values,
            total_examples=model.corpus_count, epochs=model.epochs)
        valid_tokens = data.iloc[0].tokens
        get_metrics_unobservable(
            tokens_doc_best=tokens_test_doc_best,
            tokens_doc_bad=tokens_test_doc_bad,
            tokens_valid=valid_tokens,
            model=model
        )
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    logger.logger.info('COMPLETED: train model')

    logger.logger.info('START: save')
    model.save(PATH_MODELS.joinpath('model_v1').with_suffix('.bin'))
    logger.logger.info('COMPLETED: save')


if __name__ == '__main__':
    main()
