# модуль где лежит вся обработка данных.
# тут обработка не только для запросов в тг, но и обработка датасета обучающего


import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib

import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

import logger
from loader import load_data
from myconfig import PATH_DATA, PATH_MODELS, PATH_TEMP
import texts.example_code as text_code


nlp = spacy.load("en_core_web_sm")
PATH_DATA = pathlib.Path(PATH_DATA)
PATH_MODELS = pathlib.Path(PATH_MODELS)
PATH_TEMP = pathlib.Path(PATH_TEMP)
reg_pattern = r'\`\`\`.+? \[\]\\n.+?\`\`\`\\n'
reg_pattern_code = r'\`\`\`.+\`\`\`'


def clean_code(text: str):
    text = text.lower()
    text = re.sub(pattern=r'\`\`\`.+? \[\]', string=text, repl='')
    text = re.sub(pattern=r'\\n', string=text, repl=r' ')
    # text = re.sub(pattern=r'\s+', string=text, repl=' ')
    text = re.sub(pattern=r'\`', string=text, repl='')
    text = re.sub(pattern=r'\\n', string=text, repl=r' ')
    text = re.sub(r'\W', lambda match: ' ' + match.group(0) + ' ', text)
    text = re.sub(r'\w+', lambda match: ' ' + match.group(0) + ' ', text)
    text = re.sub(pattern=r'\s+', repl=' ', string=text)
    return text


def code_tokenize(text: str):
    doc = nlp(text)
    token = [i for i in map(lambda x: f'{x.text}', list(doc)) if i != ' ']
    return token


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


def saver_data(data: pd.DataFrame, chunk_name: int):
    path_ = PATH_TEMP.joinpath(f'data_{chunk_name}.csv')
    data.to_csv(path_or_buf=path_)


def tagged_docs(i, doc):
    document = TaggedDocument(words=doc, tags=[i])
    return document


def parsing_code(data: pd.DataFrame) -> pd.DataFrame:
    # Функция для поиска
    code_data = pd.DataFrame(columns=['code'])

    for i in tqdm(range(data.shape[0])):
        post_content = data.iloc[i, data.columns.get_loc('post_content')]
        if (not post_content) or (post_content is None) or (post_content is np.nan):
            continue
        find_code = re.findall(string=post_content, pattern=reg_pattern)
        if find_code:
            code_data = pd.concat(
                [code_data, pd.DataFrame(find_code, columns=['code'])])
            continue
        find_code = re.findall(
            string=post_content, pattern=reg_pattern_code)
        if find_code:
            code_data = pd.concat(
                [code_data, pd.DataFrame(find_code, columns=['code'])])
            continue
        code_data = pd.concat(
            [code_data, pd.DataFrame([post_content], columns=['code'])])
    return code_data


def main():

    path_data = PATH_DATA.joinpath('data').with_suffix('.csv')

    chunk_size = 10000  # Размер порции данных для чтения
    for id_, chunk in enumerate(pd.read_csv(path_data,
                                            chunksize=chunk_size)):
        # Выполнение преобразований над каждой порцией данных
        logger.logger.info(f'START -- processing chunk: {id_} --')

        logger.logger.debug(f'droping chunk: {id_}')
        chunk.drop_duplicates(subset=['id_sol'], inplace=True)
        chunk.dropna(subset=['post_content'], inplace=True)
        # parsing

        logger.logger.debug(f'parsing code chunk: {id_}')
        chunk = parsing_code(data=chunk)

        chunk.reset_index(inplace=True)

        logger.logger.debug(f'clean code chunk: {id_}')
        chunk = chunk.assign(
            clean_code=chunk.code.map(clean_code))

        logger.logger.debug(f'tokenize code chunk: {id_}')
        chunk = chunk.assign(
            tokens=chunk.clean_code.map(code_tokenize))

        logger.logger.debug(f'tagged docs chunk: {id_}')
        chunk = chunk.assign(
            tagged_docs=chunk[
                ['index', 'tokens']].apply(lambda x: tagged_docs(x[0], x[1]), axis=1))

        logger.logger.debug(f'save temp chunk: {id_}')
        saver_data(data=chunk, chunk_name=id_)
        logger.logger.info(f'COMPLETED -- processing chunk: {id_} --')

    # logger.logger.info('START -- crate test data --')
    # test_doc_best = clean_code(text_code.test_doc_best)
    # tokens_test_doc_best = code_tokenize(test_doc_best)
    # test_doc_bad = clean_code(text_code.test_doc_bad)
    # tokens_test_doc_bad = code_tokenize(test_doc_bad)
    # logger.logger.info('COMPLETED -- crate test data --')

    # logger.logger.info('START -- learn model --')
    # model = Doc2Vec(
    #     vector_size=200,
    #     window=50,
    #     min_count=2,
    #     epochs=10,
    #     alpha=0.025,
    #     min_alpha=0.025)
    # model.build_vocab(list(code_data.tagged_docs.values))
    # for epoch in range(10):

    #     if epoch % 2 == 0:
    #         logger.logger.debug(f'now training epoch {epoch}\n')
    #     model.train(
    #         code_data.tagged_docs.values,
    #         total_examples=model.corpus_count, epochs=model.epochs)
    #     valid_tokens = code_data.iloc[0].tokens
    #     get_metrics_unobservable(
    #         tokens_doc_best=tokens_test_doc_best,
    #         tokens_doc_bad=tokens_test_doc_bad,
    #         tokens_valid=valid_tokens,
    #         model=model
    #     )
    #     model.alpha -= 0.002
    #     model.min_alpha = model.alpha
    # logger.logger.info('COMPLETED -- learn model --')
    # logger.logger.info('START -- save --')
    # code_data.to_pickle(PATH_DATA.joinpath('code_data').with_suffix('pickle'))
    # model.save(PATH_MODELS.joinpath('model_v1').with_suffix('.bin'))
    # logger.logger.info('COMPLETED -- save --')


if __name__ == "__main__":
    main()
