# variable_txt.py
# author Declan Fox
# tests writing variable summaries to text file.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
 
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

#split out species
df1 = df[0:50]
df2 = df[50:100]
df3 = df[100:150]

def mean(data_frame, col):
    mean = data_frame[col].mean()
    return mean

def min(data_frame, col):
    min = data_frame[col].min()
    return min

def max(data_frame, col):
    max = data_frame[col].max()
    return max




FILENAME = "variables.txt"

Sepal_length = (f'''Sepal Length:\n
The maximum Sepal length is {max(df, 'sepal length (cm)')}, the minimum is {min(df, 'sepal length (cm)')} and the mean is {round(mean(df, 'sepal length (cm)'),3)}\n
For Setosa the maximum Sepal length is {max(df1, 'sepal length (cm)')}, the minimum is {min(df1, 'sepal length (cm)')} and the mean is {round(mean(df1, 'sepal length (cm)'),3)}
For Versicolor the maximum Sepal length is {max(df2, 'sepal length (cm)')}, the minimum is {min(df2, 'sepal length (cm)')} and the mean is {round(mean(df2, 'sepal length (cm)'),3)}
For Virginica the maximum Sepal length is {max(df3, 'sepal length (cm)')}, the minimum is {min(df3, 'sepal length (cm)')} and the mean is {round(mean(df3, 'sepal length (cm)'),3)}
''')
Sepal_width = (f'''\nSepal Width:\n
The maximum Sepal Width is {max(df, 'sepal width (cm)')}, the minimum is {min(df, 'sepal width (cm)')} and the mean is {round(mean(df, 'sepal width (cm)'),3)}\n
For Setosa the maximum Sepal Width is {max(df1, 'sepal width (cm)')}, the minimum is {min(df1, 'sepal width (cm)')} and the mean is {round(mean(df1, 'sepal width (cm)'),3)}
For Versicolor the maximum Sepal Width is {max(df2, 'sepal width (cm)')}, the minimum is {min(df2, 'sepal width (cm)')} and the mean is {round(mean(df2, 'sepal width (cm)'),3)}
For Virginica the maximum Sepal Width is {max(df3, 'sepal width (cm)')}, the minimum is {min(df3, 'sepal width (cm)')} and the mean is {round(mean(df3, 'sepal width (cm)'),3)}
''')

Petal_length = (f'''\nPetal Length:
The maximum Petal length is {max(df, 'petal length (cm)')}, the minimum is {min(df, 'petal length (cm)')} and the mean is {round(mean(df, 'petal length (cm)'),3)}\n
For Setosa the maximum Petal length is {max(df1, 'petal length (cm)')}, the minimum is {min(df1, 'petal length (cm)')} and the mean is {round(mean(df1, 'petal length (cm)'),3)}
For Versicolor the maximum Petal length is {max(df2, 'petal length (cm)')}, the minimum is {min(df2, 'petal length (cm)')} and the mean is {round(mean(df2, 'petal length (cm)'),3)} 
For Virginica the maximum Petal length is {max(df3, 'petal length (cm)')}, the minimum is {min(df3, 'petal length (cm)')} and the mean is {round(mean(df3, 'petal length (cm)'),3)}
''')
Petal_width	= (f'''\nPetal Width:
The maximum Petal Width is {max(df, 'petal width (cm)')}, the minimum is {min(df, 'petal width (cm)')} and the mean is {round(mean(df, 'petal width (cm)'),3)}\n
For Setosa the maximum Petal Width is {max(df1, 'petal width (cm)')}, the minimum is {min(df1, 'petal width (cm)')} and the mean is {round(mean(df1, 'petal width (cm)'),3)}
For Versicolor the maximum Petal Width is {max(df2, 'petal width (cm)')}, the minimum is {min(df2, 'petal width (cm)')} and the mean is {round(mean(df2, 'petal width (cm)'),3)}
For Virginica the maximum Petal Width is {max(df3, 'petal width (cm)')}, the minimum is {min(df3, 'petal width (cm)')} and the mean is {round(mean(df3, 'petal width (cm)'),3)}
''')

Species = '''\nSpecies:
This will hold a summary of what the species variable is and the data it contains\n'''

with open(FILENAME, 'wt') as f:
    f.write(Sepal_length)
    f.write(Sepal_width)
    f.write(Petal_length)
    f.write(Petal_width)
    f.write(Species)