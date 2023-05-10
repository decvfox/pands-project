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

def std_dev(data_frame, col):
    std = data_frame[col].std()
    return std

def median(data_frame, col):
    median = data_frame[col].median()
    return median

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

# split training data by species
df_setosa = df_train[df_train['Species'] == 'Iris-setosa']
df_versicolor = df_train[df_train['Species'] == 'Iris-versicolor']
df_virginica = df_train[df_train['Species'] == 'Iris-virginica']

# Calculate Means
setosa_means = [mean(df_setosa, 'sepal length (cm)' ), mean(df_setosa, 'sepal width (cm)' ), 
                mean(df_setosa, 'petal length (cm)' ), mean(df_setosa, 'petal width (cm)' )]
versicolor_means = [mean(df_versicolor, 'sepal length (cm)' ), mean(df_versicolor, 'sepal width (cm)' ), 
                    mean(df_versicolor, 'petal length (cm)' ), mean(df_versicolor, 'petal width (cm)' )]
virginica_means = [mean(df_virginica, 'sepal length (cm)' ), mean(df_virginica, 'sepal width (cm)' ), 
                   mean(df_virginica, 'petal length (cm)' ), mean(df_virginica, 'petal width (cm)' )]

# calculate standard deviations	
setosa_stds = [std_dev(df_setosa, 'sepal length (cm)' ), std_dev(df_setosa, 'sepal width (cm)' ), 
                std_dev(df_setosa, 'petal length (cm)' ), std_dev(df_setosa, 'petal width (cm)' )]
versicolor_stds = [std_dev(df_versicolor, 'sepal length (cm)' ), std_dev(df_versicolor, 'sepal width (cm)' ), 
                    std_dev(df_versicolor, 'petal length (cm)' ), std_dev(df_versicolor, 'petal width (cm)' )]
virginica_stds = [std_dev(df_virginica, 'sepal length (cm)' ), std_dev(df_virginica, 'sepal width (cm)' ), 
					std_dev(df_virginica, 'petal length (cm)' ),std_dev(df_virginica, 'petal width (cm)' )]

# Calculate Medians
setosa_medians = [median(df_setosa, 'sepal length (cm)' ), median(df_setosa, 'sepal width (cm)' ), 
                median(df_setosa, 'petal length (cm)' ), median(df_setosa, 'petal width (cm)' )]
versicolor_medians = [median(df_versicolor, 'sepal length (cm)' ), median(df_versicolor, 'sepal width (cm)' ), 
                    median(df_versicolor, 'petal length (cm)' ), median(df_versicolor, 'petal width (cm)' )]
virginica_medians = [median(df_virginica, 'sepal length (cm)' ), median(df_virginica, 'sepal width (cm)' ), 
                   median(df_virginica, 'petal length (cm)' ), median(df_virginica, 'petal width (cm)' )]

# Calculate Distances from the Mean
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
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
    
    distances = [setosa_dist, versicolor_dist, virginica_dist]
    min_species = 0
    min = distances[0]
    for i in range(len(distances)):
        if distances[i] < min:
            min = distances[i]
            min_species = i
    guess = species[min_species]
    guesses.append(guess)
#add guesses column
df_test['Guesses'] = guesses

# Calculate Distances from the Mean/Standard Deviation
guesses2 = []
for index, row in df_test.iterrows():
    setosa_dist = abs(row['sepal length (cm)'] - setosa_means[0]) / setosa_stds[0]
    setosa_dist += abs(row['sepal width (cm)'] - setosa_means[1]) / setosa_stds[1]
    setosa_dist += abs(row['petal length (cm)'] - setosa_means[2]) / setosa_stds[2]
    setosa_dist += abs(row['petal width (cm)'] - setosa_means[3]) / setosa_stds[3]
    
    versicolor_dist = abs(row['sepal length (cm)'] - versicolor_means[0]) / versicolor_stds[0]
    versicolor_dist += abs(row['sepal width (cm)'] - versicolor_means[1]) / versicolor_stds[1]
    versicolor_dist += abs(row['petal length (cm)'] - versicolor_means[2])/versicolor_stds[2]
    versicolor_dist += abs(row['petal width (cm)'] - versicolor_means[3]) / versicolor_stds[3]

    virginica_dist = abs(row['sepal length (cm)'] - virginica_means[0]) / virginica_stds[0]
    virginica_dist += abs(row['sepal width (cm)'] - virginica_means[1]) / virginica_stds[1]
    virginica_dist += abs(row['petal length (cm)'] - virginica_means[2])/ virginica_stds[2]
    virginica_dist += abs(row['petal width (cm)'] - virginica_means[3]) / virginica_stds[3]

    distances = [setosa_dist, versicolor_dist, virginica_dist]
    min_species = 0
    min = distances[0]
    for i in range(len(distances)):
        if distances[i] < min:
            min = distances[i]
            min_species = i
    guess = species[min_species]
    guesses2.append(guess)
#add guesses2 column    
df_test['Guesses2'] = guesses2

# Calculate Distances from the Median
guesses3 = []
for index, row in df_test.iterrows():
    setosa_dist = abs(row['sepal length (cm)'] - setosa_medians[0])
    setosa_dist += abs(row['sepal width (cm)'] - setosa_medians[1])
    setosa_dist += abs(row['petal length (cm)'] - setosa_medians[2])
    setosa_dist += abs(row['petal width (cm)'] - setosa_medians[3])
    
    versicolor_dist = abs(row['sepal length (cm)'] - versicolor_medians[0])
    versicolor_dist += abs(row['sepal width (cm)'] - versicolor_medians[1])
    versicolor_dist += abs(row['petal length (cm)'] - versicolor_medians[2])
    versicolor_dist += abs(row['petal width (cm)'] - versicolor_medians[3])

    virginica_dist = abs(row['sepal length (cm)'] - virginica_medians[0])
    virginica_dist += abs(row['sepal width (cm)'] - virginica_medians[1])
    virginica_dist += abs(row['petal length (cm)'] - virginica_medians[2])
    virginica_dist += abs(row['petal width (cm)'] - virginica_medians[3])

    distances = [setosa_dist, versicolor_dist, virginica_dist]
    min_species = 0
    min = distances[0]
    for i in range(len(distances)):
        if distances[i] < min:
            min = distances[i]
            min_species = i
    guess = species[min_species]
    guesses3.append(guess)
#add guesses column
df_test['Guesses3'] = guesses3

# Print out results
i = 0
for index, row in df_test.iterrows():
    if row['Species'] == guesses[i]:
        guesses[i] = 'true'
    else:
        guesses[i] = 'false'
    if row['Species'] == guesses2[i]:
        guesses2[i] = 'true'
    else:
        guesses2[i] = 'false'
    if row['Species'] == guesses3[i]:
        guesses3[i] = 'true'
    else:
        guesses3[i] = 'false'

    print(f"Species =  {row['Species']}, Distance from Mean = {guesses[i]}, Distance from Mean/Std = {guesses2[i]}, Distance from Median = {guesses3[i]}")
    i += 1

df_percent = pd.DataFrame(guesses)
df_percent.columns = ['guess']
df_percent['guess2'] = guesses2
df_percent['guess3'] = guesses3

print('Dist to Mean Accuracy = ',round((df_percent.guess.value_counts().true/45)*100, 2),'\b''%')
print('Dist to Mean/ Standard Deviation Accuracy = ',round((df_percent.guess2.value_counts().true/45)*100, 2),'\b' '%')
print('Dist to Median Accuracy = ',round((df_percent.guess3.value_counts().true/45)*100, 2),'\b''%')

# write to Excel
with pd.ExcelWriter('training_data.xlsx', engine='openpyxl') as writer: 
    df_setosa.to_excel(writer, sheet_name='Setosa',index=False)
    df_versicolor.to_excel(writer, sheet_name='Versicolor',index=False)
    df_virginica.to_excel(writer, sheet_name='Virginica',index=False)
    df_test.to_excel(writer, sheet_name='test data',index=False)
    