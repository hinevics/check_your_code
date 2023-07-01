# модуль где лежит вся обработка данных.
# тут обработка не только для запросов в тг, но и обработка датасета обучающего


import re
import pandas as pd
from tqdm import tqdm

from loader import load_data
from config import PATH_DATA


reg_pattern = r'\`\`\`.+? \[\]\\n.+?\`\`\`\\n'
reg_pattern_code = r'\`\`\`.+\`\`\`'


def main():
    data = load_data(PATH_DATA)
    code_data = pd.DataFrame(columns=['code'])
    for i in tqdm(range(data.shape[0])):
        post_content = data.iloc[i, data.columns.get_loc('post_content')]
        find_code = re.findall(string=post_content, pattern=reg_pattern)
        if find_code:
            code_data = pd.concat([code_data, pd.DataFrame(find_code, columns=['code'])])
            continue
        find_code = re.findall(string=post_content, pattern=reg_pattern_code)
        if find_code:
            code_data = pd.concat([code_data, pd.DataFrame(find_code, columns=['code'])])
            continue
        code_data = pd.concat([code_data, pd.DataFrame([post_content], columns=['code'])])
    code_data.reset_index(inplace=True)


if __name__ == "__main__":
    main()
