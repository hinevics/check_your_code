# Модуль для processing of data
import re


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


def code_tokenize(text: str, clean: bool = True) -> list:
    text: str = clean_code(text) if clean else text
    docs: list = text.split(' ')
    token: list = [i for i in docs if i not in [' ', '']]
    return token
