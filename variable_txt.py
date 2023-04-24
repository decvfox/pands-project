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

def mean(col):
    mean = df[col].mean()
    return mean

def min(col):
    min = df[col].min()
    return min

def max(col):
    max = df[col].max()
    return max




FILENAME = "variables.txt"

Sepal_length = (f'''Sepal Length:
The maximum Sepal length is {max('sepal length (cm)')}, the minimum is {min('sepal length (cm)')} and the mean is {round(mean('sepal length (cm)'),3)}\n''')
Sepal_width = (f'''\nSepal Width:
The maximum Sepal Width is {max('sepal width (cm)')}, the minimum is {min('sepal width (cm)')} and the mean is {round(mean('sepal width (cm)'),3)}\n''')
Petal_length = (f'''\nPetal Length:
The maximum Petal length is {max('petal length (cm)')}, the minimum is {min('petal length (cm)')} and the mean is {round(mean('petal length (cm)'),3)}\n''')
Petal_width	= (f'''\nPetal Width:
The maximum Petal Width is {max('petal width (cm)')}, the minimum is {min('petal width (cm)')} and the mean is {round(mean('petal width (cm)'),3)}\n''')
Species = '''\nSpecies:
This will hold a summary of what the species variable is and the data it contains\n'''

with open(FILENAME, 'wt') as f:
    f.write(Sepal_length)
    f.write(Sepal_width)
    f.write(Petal_length)
    f.write(Petal_width)
    f.write(Species)