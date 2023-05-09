# Split_70_30.py
# Author: Declan Fox
# split a dataframe up randomly 70:30


import csv
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import requests
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
import seaborn as sns

# import from CSV file
FILENAME="iris_data.csv"
with open(FILENAME, "rt") as file:
    csvReader = csv.reader(file, delimiter = ',') 

# load to Pandas Dataframe
df1 = pd.read_csv('iris_data.csv', header=None)
# add column names
df1.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'Species' ]


# Download from API
iris = load_iris()
df2= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
# Rename target column to species
df2.rename(columns={'target': 'Species'}, inplace=True)
# replace numbers with species' names
df2['Species'].replace({ 0 : 'Iris-setosa', 1 : 'Iris-versicolor', 2 : 'Iris-virginica'}, inplace=True)


# Scrape from website
py_url = "https://en.wikipedia.org/wiki/Iris_flower_data_set"
py_page = requests.get (py_url)
py_soup = bs(py_page.text, 'html.parser')
py_table = py_soup.find ('table', {'class':'wikitable'})
py_rows = py_table.find_all ('tr')

df3=pd.read_html(str(py_table))
# convert list to dataframe
df3=pd.DataFrame(df3[0])
# Rename columns
df3.rename(columns={'Sepal length': 'sepal length (cm)', 'Sepal width': 'sepal width (cm)'}, inplace=True)
df3.rename(columns={'Petal length': 'petal length (cm)', 'Petal width': 'petal width (cm)'}, inplace=True)
# update species' names to match the other dataframes
df3['Species'].replace({ 'I. setosa' : 'Iris-setosa', 'I. versicolor' : 'Iris-versicolor', 'I. virginica' : 'Iris-virginica'}, inplace=True)
# Delete Dataset order column
df3 = df3.drop('Dataset order', axis=1)

# write to Excel
with pd.ExcelWriter('iris_dataset.xlsx', engine='openpyxl') as writer: 
    df1.to_excel(writer, sheet_name='CSV',index=False)
    df2.to_excel(writer, sheet_name='API',index=False)
    df3.to_excel(writer, sheet_name='Wiki',index=False)

df = df2

df_train = df.sample(frac = 0.70)
df_test = df.drop(df_train.index)

print(df_train)
print(df_test)