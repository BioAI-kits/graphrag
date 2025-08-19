import os

import numpy as np
import pandas as pd
import csv

'''
把output中所有.parquet的文件转换为csv
'''

parquet_dir = '../graphrag_zh/output'
csv_dir = r'export'
os.makedirs(csv_dir, exist_ok=True)

def clean_quotes(value):
    """
       清理并格式化字符串中的引号。
       该函数旨在处理字符串数据中的引号问题，确保字符串适合于进一步的数据处理或存储。
       如果输入不是字符串，则直接返回，不进行处理。

       参数:
       value: 任意类型。这是待处理的输入值，预期为字符串。

       返回:
       处理后的字符串，或者原值，如果输入不是字符串的话。
       """

    # TZX 添加，部分数据类型为ndarray，需要转换为字符串
    if isinstance(value, np.ndarray):
        value = ','.join(map(str, value))

    # 检查输入是否为字符串类型
    if isinstance(value, str):
        # 移除字符串首尾的空格，以及内部多余的引号
        value = value.strip().replace('""', '"').replace('"', '')
        value = value.replace('\n', '')   # TZX添加
        # print(value)

        # 如果字符串内包含逗号或引号，则在字符串首尾添加引号
        if ',' in value or '"' in value:
            value = f'"{value}"'

    # 返回处理后的值
    return value


for file_name in os.listdir(parquet_dir):
    if file_name.endswith('.parquet'):
        parquet_file = os.path.join(parquet_dir, file_name)
        print(parquet_file)
        csv_file = os.path.join(csv_dir, file_name.replace('.parquet', '.csv'))

        df = pd.read_parquet(parquet_file)

        # for column in df.select_dtypes(include=['object']).columns:
        #     df[column] = df[column].apply(clean_quotes)

        df.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f'数据{parquet_file} to {csv_file} successfull')
print('All parquet files have been converted to CSV.')
