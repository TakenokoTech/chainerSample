import pandas as pd
import numpy as np

'''
# Series
python_list = [1, 3, 5, np.nan, None, -1]
pandas_series = pd.Series(python_list)
print(python_list)
print(pandas_series)

# DataFrame
python_list = [['alice', 100],['bob', 90], ['charlie', 85]]
pandas_dataframe = pd.DataFrame(python_list, columns=['name', 'score'])
print(pandas_dataframe)

# Series -> DataFrame
names_series = pd.Series(['alice', 'bob', 'charlie'])
scores_series = pd.Series([100, 90, 85])
pandas_dataframe = pd.concat([names_series, scores_series], axis=1)
print(pandas_dataframe)

# 
names = [
    {'id': 0, 'name': 'alice', 'age': 21},
    {'id': 1, 'name': 'bob', 'age': 24},
    {'id': 2, 'name': 'charlie', 'age': 22},
    {'id': 4, 'name': 'dave', 'age': None}
]
names_dataframe = pd.DataFrame(names)
scores = [
    {'id': 0, 'score': 100, 'retest': True},
    {'id': 1, 'score': 90, 'retest': False},
    {'id': 2, 'score': 85, 'retest': True},
    {'id': 3, 'score': 0, 'retest': False}
]
scores_dataframe = pd.DataFrame(scores)
merge = pd.merge(names_dataframe, scores_dataframe, how='right', on='id')
print(merge)
'''

dataframe = pd.DataFrame(np.random.random_sample(80).reshape((20, 4)), columns=['A', 'B', 'C', 'D'])
print(dataframe)
print(dataframe.where(dataframe.A > 0.5).dropna())
