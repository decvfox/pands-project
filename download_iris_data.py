# Download the data
# Author: Declan Fox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from sklearn.datasets import load_iris
iris = load_iris()
 
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
 
# select setosa and versicolor
y = df.iloc[0:150, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
 
# extract sepal length and petal length
X = df.iloc[0:150, [0, 2]].values
 
# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='blue', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='green', marker='s', label='Versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1],
            color='red', marker='^', label='Virginica')
 
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
 
# plt.savefig('images/02_06.png', dpi=300)
plt.show()

