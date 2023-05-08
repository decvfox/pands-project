# analysis.py
#Author Declan Fox
# place holder for final script

import csv
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import requests
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
import seaborn as sns

# import UCI dataset from CSV file
FILENAME="iris_data.csv"
with open(FILENAME, "rt") as file:
    csvReader = csv.reader(file, delimiter = ',') 

# load to Pandas dataframe
df_uci = pd.read_csv('iris_data.csv', header=None)
# add column names
df_uci.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'Species' ]

# Download from API
iris = load_iris()
df_api= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
# Rename target column to species
df_api.rename(columns={'target': 'Species'}, inplace=True)
# replace numbers with species' names
df_api['Species'].replace({ 0 : 'Iris-setosa', 1 : 'Iris-versicolor', 2 : 'Iris-virginica'}, inplace=True)


# Scrape from website
py_url = "https://en.wikipedia.org/wiki/Iris_flower_data_set"
py_page = requests.get (py_url)
py_soup = bs(py_page.text, 'html.parser')
py_table = py_soup.find ('table', {'class':'wikitable'})
py_rows = py_table.find_all ('tr')

df_wiki=pd.read_html(str(py_table))
# convert list to dataframe
df_wiki=pd.DataFrame(df_wiki[0])
# Rename columns
df_wiki.rename(columns={'Sepal length': 'sepal length (cm)', 'Sepal width': 'sepal width (cm)'}, inplace=True)
df_wiki.rename(columns={'Petal length': 'petal length (cm)', 'Petal width': 'petal width (cm)'}, inplace=True)
# update species' names to match the other dataframes
df_wiki['Species'].replace({ 'I. setosa' : 'Iris-setosa', 'I. versicolor' : 'Iris-versicolor', 'I. virginica' : 'Iris-virginica'}, inplace=True)
# Delete Dataset order column
df_wiki = df_wiki.drop('Dataset order', axis=1)

# write to Excel
with pd.ExcelWriter('iris_datasets.xlsx', engine='openpyxl') as writer: 
    df_uci.to_excel(writer, sheet_name='UCI Dataset',index=False)
    df_api.to_excel(writer, sheet_name='SK Learn Dataset',index=False)
    df_wiki.to_excel(writer, sheet_name='Wikipedia Dataset',index=False)

#split out species
df1 = df_api[0:50]
df2 = df_api[50:100]
df3 = df_api[100:150]

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
The maximum Sepal length is {max(df_api, 'sepal length (cm)')}, the minimum is {min(df_api, 'sepal length (cm)')} and the mean is {round(mean(df_api, 'sepal length (cm)'),3)}\n
For Setosa the maximum Sepal length is {max(df1, 'sepal length (cm)')}, the minimum is {min(df1, 'sepal length (cm)')} and the mean is {round(mean(df1, 'sepal length (cm)'),3)}
For Versicolor the maximum Sepal length is {max(df2, 'sepal length (cm)')}, the minimum is {min(df2, 'sepal length (cm)')} and the mean is {round(mean(df2, 'sepal length (cm)'),3)}
For Virginica the maximum Sepal length is {max(df3, 'sepal length (cm)')}, the minimum is {min(df3, 'sepal length (cm)')} and the mean is {round(mean(df3, 'sepal length (cm)'),3)}
''')
Sepal_width = (f'''\nSepal Width:\n
The maximum Sepal Width is {max(df_api, 'sepal width (cm)')}, the minimum is {min(df_api, 'sepal width (cm)')} and the mean is {round(mean(df_api, 'sepal width (cm)'),3)}\n
For Setosa the maximum Sepal Width is {max(df1, 'sepal width (cm)')}, the minimum is {min(df1, 'sepal width (cm)')} and the mean is {round(mean(df1, 'sepal width (cm)'),3)}
For Versicolor the maximum Sepal Width is {max(df2, 'sepal width (cm)')}, the minimum is {min(df2, 'sepal width (cm)')} and the mean is {round(mean(df2, 'sepal width (cm)'),3)}
For Virginica the maximum Sepal Width is {max(df3, 'sepal width (cm)')}, the minimum is {min(df3, 'sepal width (cm)')} and the mean is {round(mean(df3, 'sepal width (cm)'),3)}
''')

Petal_length = (f'''\nPetal Length:
The maximum Petal length is {max(df_api, 'petal length (cm)')}, the minimum is {min(df_api, 'petal length (cm)')} and the mean is {round(mean(df_api, 'petal length (cm)'),3)}\n
For Setosa the maximum Petal length is {max(df1, 'petal length (cm)')}, the minimum is {min(df1, 'petal length (cm)')} and the mean is {round(mean(df1, 'petal length (cm)'),3)}
For Versicolor the maximum Petal length is {max(df2, 'petal length (cm)')}, the minimum is {min(df2, 'petal length (cm)')} and the mean is {round(mean(df2, 'petal length (cm)'),3)} 
For Virginica the maximum Petal length is {max(df3, 'petal length (cm)')}, the minimum is {min(df3, 'petal length (cm)')} and the mean is {round(mean(df3, 'petal length (cm)'),3)}
''')
Petal_width	= (f'''\nPetal Width:
The maximum Petal Width is {max(df_api, 'petal width (cm)')}, the minimum is {min(df_api, 'petal width (cm)')} and the mean is {round(mean(df_api, 'petal width (cm)'),3)}\n
For Setosa the maximum Petal Width is {max(df1, 'petal width (cm)')}, the minimum is {min(df1, 'petal width (cm)')} and the mean is {round(mean(df1, 'petal width (cm)'),3)}
For Versicolor the maximum Petal Width is {max(df2, 'petal width (cm)')}, the minimum is {min(df2, 'petal width (cm)')} and the mean is {round(mean(df2, 'petal width (cm)'),3)}
For Virginica the maximum Petal Width is {max(df3, 'petal width (cm)')}, the minimum is {min(df3, 'petal width (cm)')} and the mean is {round(mean(df3, 'petal width (cm)'),3)}
''')

Species = '''\nSpecies:
There are 3 species of Iris flower, with 50 sets of data for each species.\n'''

with open(FILENAME, 'wt') as f:
    f.write(Sepal_length)
    f.write(Sepal_width)
    f.write(Petal_length)
    f.write(Petal_width)
    f.write(Species)

    # Compare Datasets
print('UCI Dataset vs SK Learn Dataset\n', df_uci.compare(df_api))
print('\nUCI Dataset vs Wikipedia Dataset\n', df_uci.compare(df_wiki))
print('\nSK Learn Dataset vs Wikipedia Dataset\n', df_api.compare(df_wiki))

# plot histogram

fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(ncols=2, nrows=2,  figsize=(10, 10))

df_api.plot(kind = 'hist', y = 'sepal length (cm)', color='c', alpha = 0.5, label = 'Sepal Length', ax=ax)
df_api.plot(kind = 'hist', y = 'sepal width (cm)', color='m', alpha = 0.5, label = 'Sepal Width', ax=ax1)
df_api.plot(kind = 'hist', y = 'petal length (cm)', color='y', alpha = 0.5, label = 'Petal Length', ax=ax2)
df_api.plot(kind = 'hist', y = 'petal width (cm)', color='k', alpha = 0.5, label = 'Petal Width', ax=ax3)

plt.savefig('histogram.png')

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

plt.savefig('histogram_species.png')

# plot scatter

fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(10, 4))

df1.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', label = 'Setosa', color='c', ax=ax)
df2.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', color='m', label = 'Versicolor', ax=ax)
df3.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', color='y', label = 'Virginica', ax=ax)
df1.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', label = 'Setosa', color='c', ax=ax1)
df2.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', color='m', label = 'Versicolor', ax=ax1)
df3.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', color='y', label = 'Virginica', ax=ax1)

# Plot Heat Map

df1_data_cols = df1.drop('Species', axis=1)
df2_data_cols = df2.drop('Species', axis=1)
df3_data_cols = df3.drop('Species', axis=1)

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

plt.savefig('heat_map.png')

# Split into training and test dataframes
df_train = df_api.sample(frac = 0.70)
df_test = df_api.drop(df_train.index)

# write to Excel
with pd.ExcelWriter('iris_datasets.xlsx', engine='openpyxl', mode="a") as writer: 
    df_train.to_excel(writer, sheet_name='Training Data',index=False)
    df_test.to_excel(writer, sheet_name='Test Data',index=False)
