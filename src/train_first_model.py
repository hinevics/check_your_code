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
from myconfig import PATH_DATA, PATH_MODELS
import texts.example_code as text_code


nlp = spacy.load("en_core_web_sm")
PATH_DATA = pathlib.Path(PATH_DATA)
PATH_MODELS = pathlib.Path(PATH_MODELS)
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


def tagged_docs(i, doc):
    document = TaggedDocument(words=doc, tags=[i])
    return document


def main():
    logger.logger.info('START -- load data --')
    data = load_data(PATH_DATA.joinpath('data').with_suffix('.csv'))
    logger.logger.info('COMPLETED -- load data --')

    logger.logger.info('START -- parsing code --')
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
    logger.logger.info('COMPLETED -- parsing code --')

    code_data.reset_index(inplace=True)

    logger.logger.info('START -- clean code --')
    code_data = code_data.assign(
        clean_code=code_data.code.map(clean_code))
    logger.logger.info('COMPLETED -- clean code --')

    logger.logger.info('START -- tokenize code --')
    code_data = code_data.assign(
        tokens=code_data.clean_code.map(code_tokenize))
    logger.logger.info('COMPLETED -- tokenize code --')

    logger.logger.info('START -- tagged docs --')
    code_data = code_data.assign(
        tagged_docs=code_data[
            ['index', 'tokens']].apply(lambda x: tagged_docs(x[0], x[1]), axis=1))
    logger.logger.info('COMPLETED -- tagged docs --')

    logger.logger.info('START -- crate test data --')
    test_doc_best = clean_code(text_code.test_doc_best)
    tokens_test_doc_best = code_tokenize(test_doc_best)
    test_doc_bad = clean_code(text_code.test_doc_bad)
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
    model.build_vocab(list(code_data.tagged_docs.values))
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
    logger.logger.info('COMPLETED -- learn model --')
    logger.logger.info('START -- save --')
    code_data.to_pickle(PATH_DATA.joinpath('code_data').with_suffix('pickle'))
    model.save(PATH_MODELS.joinpath('model_v1').with_suffix('.bin'))
    logger.logger.info('COMPLETED -- save --')


if __name__ == "__main__":
    main()
