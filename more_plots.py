# more_plots.py
# Author: Declan Fox
# tests plotting
  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns

iris = load_iris()
 
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
 
# plot heat map


#split out species
df1 = df[0:50]
df2 = df[50:100]
df3 = df[100:150]

df1_data_cols = df1.drop('target', axis=1)
df2_data_cols = df2.drop('target', axis=1)
df3_data_cols = df3.drop('target', axis=1)

fig, ((ax, ax1, ax2)) = plt.subplots(ncols=3, nrows=1,  figsize=(8, 6))

sns.heatmap(df1_data_cols, cmap ='tab20_r', vmin=0, vmax=10, ax=ax)
ax.set_title('Setosa')
plt.tight_layout()

sns.heatmap(df2_data_cols, cmap ='tab20_r', vmin=0, vmax=10, ax=ax1)
ax1.set_title('Versicolor')
plt.tight_layout()

sns.heatmap(df3_data_cols, cmap ='tab20_r', vmin=0, vmax=10, ax=ax2)
ax2.set_title('Virginica')
plt.tight_layout()

plt.show()
