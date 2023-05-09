# messing_with_means.py
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

def mean(data_frame, col):
    mean = data_frame[col].mean()
    return mean

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

df = df2

df_train = df.sample(frac = 0.70)
df_test = df.drop(df_train.index)

# split df_train by species

df_setosa = df_train[df_train['Species'] == 'Iris-setosa']
df_versicolor = df_train[df_train['Species'] == 'Iris-versicolor']
df_virginica = df_train[df_train['Species'] == 'Iris-virginica']

setosa_means = [mean(df_setosa, 'sepal length (cm)' ), mean(df_setosa, 'sepal width (cm)' ), 
                mean(df_setosa, 'petal length (cm)' ), mean(df_setosa, 'petal width (cm)' )]
versicolor_means = [mean(df_versicolor, 'sepal length (cm)' ), mean(df_versicolor, 'sepal width (cm)' ), 
                    mean(df_versicolor, 'petal length (cm)' ), mean(df_versicolor, 'petal width (cm)' )]
virginica_means = [mean(df_virginica, 'sepal length (cm)' ), mean(df_virginica, 'sepal width (cm)' ), 
                   mean(df_virginica, 'petal length (cm)' ), mean(df_virginica, 'petal width (cm)' )]

#print(setosa_means)
#print(versicolor_means)
#print(virginica_means)
guesses = []
for index, row in df_test.iterrows():
    setosa_dist = abs(row['sepal length (cm)'] - setosa_means[0])
    setosa_dist += abs(row['sepal width (cm)'] - setosa_means[1])
    setosa_dist += abs(row['petal length (cm)'] - setosa_means[2])
    setosa_dist += abs(row['petal width (cm)'] - setosa_means[3])
    
    versicolor_dist = abs(row['sepal length (cm)'] - versicolor_means[0])
    versicolor_dist += abs(row['sepal width (cm)'] - versicolor_means[1])
    versicolor_dist += abs(row['petal length (cm)'] - versicolor_means[2])
    versicolor_dist += abs(row['petal width (cm)'] - versicolor_means[3])

    virginica_dist = abs(row['sepal length (cm)'] - virginica_means[0])
    virginica_dist += abs(row['sepal width (cm)'] - virginica_means[1])
    virginica_dist += abs(row['petal length (cm)'] - virginica_means[2])
    virginica_dist += abs(row['petal width (cm)'] - virginica_means[3])

    print(setosa_dist, versicolor_dist, virginica_dist, )
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    distances = [setosa_dist, versicolor_dist, virginica_dist]
    min_species = 0
    min = distances[0]
    for i in range(len(distances)):
        if distances[i] < min:
            min = distances[i]
            min_species = i
    guess = species[min_species]
    guesses.append(guess)
print(guesses)
df_test['Guesses'] = guesses

# write to Excel
with pd.ExcelWriter('training_data.xlsx', engine='openpyxl') as writer: 
    df_setosa.to_excel(writer, sheet_name='Setosa',index=False)
    df_versicolor.to_excel(writer, sheet_name='Versicolor',index=False)
    df_virginica.to_excel(writer, sheet_name='Virginica',index=False)
    df_test.to_excel(writer, sheet_name='test data',index=False)