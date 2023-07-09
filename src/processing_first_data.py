# модуль где лежит вся обработка данных.
# тут обработка не только для запросов в тг, но и обработка датасета обучающего


import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib

from gensim.models.doc2vec import TaggedDocument

import logger
from myconfig import PATH_DATA, PATH_MODELS, PATH_TEMP


PATH_DATA = pathlib.Path(PATH_DATA)
PATH_MODELS = pathlib.Path(PATH_MODELS)
PATH_TEMP = pathlib.Path(PATH_TEMP)
reg_pattern = r'\`\`\`.+? \[\]\\n.+?\`\`\`\\n'
reg_pattern_code = r'\`\`\`.+\`\`\`'


def clean_code(text: str, patter_type: bool = True):
    text = text.lower()
    text = re.sub(pattern=r'\\n', string=text, repl=r' ')
    # text = re.sub(pattern=r'\s+', string=text, repl=' ')
    text = re.sub(pattern=r'\`', string=text, repl='')
    text = re.sub(pattern=r'\\n', string=text, repl=r' ')
    text = re.sub(r'\W', lambda match: ' ' + match.group(0) + ' ', text)
    text = re.sub(r'\w+', lambda match: ' ' + match.group(0) + ' ', text)
    text = re.sub(pattern=r'\s+', repl=' ', string=text)
    return text


def code_tokenize(text: str, clean: bool = True, patter_type: bool = True) -> list:
    text: str = clean_code(text, patter_type=patter_type) if clean else text
    docs: list = text.split(' ')
    token: list = [
        i for i in docs if i not in [' ', '', 'swift', 'python', 'java', 'c++', 'cpp']
    ]
    return token


def saver_data(data: pd.DataFrame, chunk_name: int):
    path_ = PATH_TEMP.joinpath(f'data_{chunk_name}.csv')
    data.to_csv(path_or_buf=path_)


def tagged_docs(i, doc):
    document = TaggedDocument(words=doc, tags=[i])
    return document


def parsing_code(data: pd.DataFrame) -> pd.DataFrame:
    # Функция для поиска
    code_data = pd.DataFrame(columns=['code', 'tokens'])

    for i in tqdm(range(data.shape[0])):
        post_content = data.iloc[i, data.columns.get_loc('post_content')]
        if (not post_content) or (post_content is None) or (post_content is np.nan):
            continue
        find_code = re.findall(string=post_content, pattern=reg_pattern)
        if find_code:
            code_and_tokens = [[c, code_tokenize(text=c)] for c in find_code]
            temp_code_data = pd.DataFrame(
                data=code_and_tokens,
                columns=['code', 'tokens'])
            code_data = pd.concat(
                [code_data, temp_code_data])
        find_code = re.findall(
            string=post_content, pattern=reg_pattern_code)
        if find_code:
            code_and_tokens = [
                [c, code_tokenize(text=c, patter_type=False)] for c in find_code]
            temp_code_data = pd.DataFrame(
                data=code_and_tokens,
                columns=['code', 'tokens'])
            code_data = pd.concat(
                [code_data, temp_code_data])
        tokens = code_tokenize(post_content, clean=True)
        temp_code_data = pd.DataFrame(
            data=[[post_content, tokens]],
            columns=['code', 'tokens']
        )
        code_data = pd.concat(
            [code_data, temp_code_data])
    return code_data


def main():

    path_data = PATH_DATA.joinpath('data').with_suffix('.csv')
    chunk_size = 10_000  # Размер порции данных для чтения
    all_chunks = pd.read_csv(path_data, chunksize=chunk_size)
    for id_, chunk in enumerate(all_chunks):
        # Выполнение преобразований над каждой порцией данных
        logger.logger.info(f'START: processing chunk: {id_}')

        logger.logger.debug(f'droping chunk: {id_}')
        chunk.drop_duplicates(subset=['id_sol'], inplace=True)
        chunk.dropna(subset=['post_content'], inplace=True)
        # parsing

        logger.logger.debug(f'parsing code chunk: {id_}')
        chunk = parsing_code(data=chunk)

        chunk.reset_index(inplace=True)
        logger.logger.debug(f'tagged docs chunk: {id_}')
        chunk = chunk.assign(
            tagged_docs=chunk[
                ['index', 'tokens']].apply(lambda x: tagged_docs(x[0], x[1]), axis=1))

        logger.logger.debug(f'save temp chunk: {id_}')
        saver_data(data=chunk, chunk_name=id_)
        logger.logger.info(f'COMPLETED: processing chunk: {id_}')


if __name__ == "__main__":
    main()
