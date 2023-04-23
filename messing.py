# Download the data
# Author: Declan Fox
# for testing concepts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
 
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

df['sep lxw'] = df['sepal length (cm)'] + df['petal length (cm)']
df['pet lxw'] = df['sepal width (cm)'] + df['petal width (cm)']
 
df['ratio'] = df['pet lxw']/df['sep lxw']*100

with pd.ExcelWriter('messing.xlsx', engine='openpyxl') as writer: 
    df.to_excel(writer, sheet_name='ratio',index=False)

print(df)

df1 = df[0:50]
df2 = df[50:100]
df3 = df[100:150]

fig, (ax) = plt.subplots(ncols=1, figsize=(5, 4))

df1.plot(kind = 'hist',  y = 'ratio', color='c', label = 'Setosa', ax=ax)
df2.plot(kind = 'hist',  y = 'ratio', color='m', label = 'Versicolor', ax=ax)
df3.plot(kind = 'hist',  y = 'ratio', color='y', label = 'Virginica', ax=ax)

plt.show()