# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('ZJ.csv')

# 获取列名
columns = df.columns.tolist()

# 将每列数据添加到列表中
data_lists = []
for column in columns:
    data_lists.append(df[column].tolist())

# 打印结果
for i, data_list in enumerate(data_lists):
    print(f'Column {i + 1}: {data_list}')
