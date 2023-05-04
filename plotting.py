# plotting.py
# Author: Declan Fox
# tests plotting
  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
 
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
 
# plot histogram

fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(ncols=2, nrows=2,  figsize=(10, 10))

df.plot(kind = 'hist', y = 'sepal length (cm)', color='c', alpha = 0.5, label = 'Sepal Length', ax=ax)
df.plot(kind = 'hist', y = 'sepal width (cm)', color='m', alpha = 0.5, label = 'Sepal Width', ax=ax1)
df.plot(kind = 'hist', y = 'petal length (cm)', color='y', alpha = 0.5, label = 'Petal Length', ax=ax2)
df.plot(kind = 'hist', y = 'petal width (cm)', color='k', alpha = 0.5, label = 'Petal Width', ax=ax3)

plt.show()# plt.savefig('histogram.png')

#split out species
df1 = df[0:50]
df2 = df[50:100]
df3 = df[100:150]

# plot histogram coloured by species

fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(ncols=2, nrows=2,  figsize=(10, 10))

df1.plot(kind = 'hist', y = 'sepal length (cm)', color='c', alpha = 0.5, label = 'Sepal Length Setosa', ax=ax)
df2.plot(kind = 'hist', y = 'sepal length (cm)', color='m', alpha = 0.5, label = 'Sepal Length Versicolor', ax=ax)
df3.plot(kind = 'hist', y = 'sepal length (cm)', color='y', alpha = 0.5, label = 'Sepal Length Virginica', ax=ax)
df1.plot(kind = 'hist', y = 'sepal width (cm)', color='c', alpha = 0.5, label = 'Sepal Width Setosa', ax=ax1)
df2.plot(kind = 'hist', y = 'sepal width (cm)', color='m', alpha = 0.5, label = 'Sepal Width Versicolor', ax=ax1)
df3.plot(kind = 'hist', y = 'sepal width (cm)', color='y', alpha = 0.5, label = 'Sepal Width Virginica', ax=ax1)
df1.plot(kind = 'hist', y = 'petal length (cm)', color='c', alpha = 0.5, label = 'Petal Length Setosa', ax=ax2)
df2.plot(kind = 'hist', y = 'petal length (cm)', color='m', alpha = 0.5, label = 'Petal Length Versicolor', ax=ax2)
df3.plot(kind = 'hist', y = 'petal length (cm)', color='y', alpha = 0.5, label = 'Petal Length Virginica', ax=ax2)
df1.plot(kind = 'hist', y = 'petal width (cm)', color='c', alpha = 0.5, label = 'Petal Width Setosa', ax=ax3)
df2.plot(kind = 'hist', y = 'petal width (cm)', color='m', alpha = 0.5, label = 'Petal Width Versicolor', ax=ax3)
df3.plot(kind = 'hist', y = 'petal width (cm)', color='y', alpha = 0.5, label = 'Petal Width Virginica', ax=ax3)

plt.show() #plt.savefig('histogram_species.png')

# plot scatter

fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(10, 4))

df1.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', label = 'Setosa', color='c', ax=ax)
df2.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', color='m', label = 'Versicolor', ax=ax)
df3.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', color='y', label = 'Virginica', ax=ax)
df1.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', label = 'Setosa', color='c', ax=ax1)
df2.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', color='m', label = 'Versicolor', ax=ax1)
df3.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', color='y', label = 'Virginica', ax=ax1)

plt.show() #plt.savefig('scatter.png')

