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
import datetime

def mean(data_frame, col):
    mean = data_frame[col].mean()
    return mean

def min(data_frame, col):
    min = data_frame[col].min()
    return min

def max(data_frame, col):
    max = data_frame[col].max()
    return max

def std_dev(data_frame, col):
    std = data_frame[col].std()
    return std

def median(data_frame, col):
    median = data_frame[col].median()
    return median

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

with open(FILENAME, 'wt') as f:
    f.write(Sepal_length)
    f.write(Sepal_width)
    f.write(Petal_length)
    f.write(Petal_width)

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

# scatter plot
fig, ((ax, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(ncols=2, nrows=3, figsize=(10, 15))

df1.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', label = 'Setosa', color='c', ax=ax)
df2.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', color='m', label = 'Versicolor', ax=ax)
df3.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'sepal width (cm)', color='y', label = 'Virginica', ax=ax)
df1.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', label = 'Setosa', color='c', ax=ax1)
df2.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', color='m', label = 'Versicolor', ax=ax1)
df3.plot(kind = 'scatter', x = 'petal length (cm)', y = 'petal width (cm)', color='y', label = 'Virginica', ax=ax1)
df1.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'petal length (cm)', label = 'Setosa', color='c', ax=ax2)
df2.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'petal length (cm)', color='m', label = 'Versicolor', ax=ax2)
df3.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'petal length (cm)', color='y', label = 'Virginica', ax=ax2)
df1.plot(kind = 'scatter', x = 'petal length (cm)', y = 'sepal width (cm)', label = 'Setosa', color='c', ax=ax3)
df2.plot(kind = 'scatter', x = 'petal length (cm)', y = 'sepal width (cm)', color='m', label = 'Versicolor', ax=ax3)
df3.plot(kind = 'scatter', x = 'petal length (cm)', y = 'sepal width (cm)', color='y', label = 'Virginica', ax=ax3)
df1.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'petal width (cm)', label = 'Setosa', color='c', ax=ax4)
df2.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'petal width (cm)', color='m', label = 'Versicolor', ax=ax4)
df3.plot(kind = 'scatter', x = 'sepal length (cm)', y = 'petal width (cm)', color='y', label = 'Virginica', ax=ax4)
df1.plot(kind = 'scatter', x = 'petal width (cm)', y = 'sepal width (cm)', label = 'Setosa', color='c', ax=ax5)
df2.plot(kind = 'scatter', x = 'petal width (cm)', y = 'sepal width (cm)', color='m', label = 'Versicolor', ax=ax5)
df3.plot(kind = 'scatter', x = 'petal width (cm)', y = 'sepal width (cm)', color='y', label = 'Virginica', ax=ax5)

plt.savefig('scatter.png')

# Plot Heat Map
# we don't need the species column for this
df1_data_cols = df1.drop('Species', axis=1)
df2_data_cols = df2.drop('Species', axis=1)
df3_data_cols = df3.drop('Species', axis=1)

fig, ((ax, ax1, ax2)) = plt.subplots(ncols=3, nrows=1,  figsize=(8, 6))
# we will use Seaborn for this plot
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

# split training data into species dataframes
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

# Calculate Distances from the Mean for each variable of each species
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
mean_guesses = []
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
    
    # sort distances add the shortest to a list
    distances = [setosa_dist, versicolor_dist, virginica_dist]
    min_species = 0
    min = distances[0]
    for i in range(len(distances)):
        if distances[i] < min:
            min = distances[i]
            min_species = i
    guess = species[min_species]
    mean_guesses.append(guess)
#add column to test dataframe for distance to mean
df_test['Mean only'] = mean_guesses

# calculate standard deviations	
setosa_stds = [std_dev(df_setosa, 'sepal length (cm)' ), std_dev(df_setosa, 'sepal width (cm)' ), 
                std_dev(df_setosa, 'petal length (cm)' ), std_dev(df_setosa, 'petal width (cm)' )]
versicolor_stds = [std_dev(df_versicolor, 'sepal length (cm)' ), std_dev(df_versicolor, 'sepal width (cm)' ), 
                    std_dev(df_versicolor, 'petal length (cm)' ), std_dev(df_versicolor, 'petal width (cm)' )]
virginica_stds = [std_dev(df_virginica, 'sepal length (cm)' ), std_dev(df_virginica, 'sepal width (cm)' ), 
					std_dev(df_virginica, 'petal length (cm)' ),std_dev(df_virginica, 'petal width (cm)' )]

# Calculate Distances from the Mean/Standard Deviation
mean_sd_guesses = []
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
    mean_sd_guesses.append(guess)
df_test['Mean/SD'] = mean_sd_guesses

# Calculate Medians
setosa_medians = [median(df_setosa, 'sepal length (cm)' ), median(df_setosa, 'sepal width (cm)' ), 
                median(df_setosa, 'petal length (cm)' ), median(df_setosa, 'petal width (cm)' )]
versicolor_medians = [median(df_versicolor, 'sepal length (cm)' ), median(df_versicolor, 'sepal width (cm)' ), 
                    median(df_versicolor, 'petal length (cm)' ), median(df_versicolor, 'petal width (cm)' )]
virginica_medians = [median(df_virginica, 'sepal length (cm)' ), median(df_virginica, 'sepal width (cm)' ), 
                   median(df_virginica, 'petal length (cm)' ), median(df_virginica, 'petal width (cm)' )]

# Calculate Distances from the Median
median_guesses = []
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
    median_guesses.append(guess)
df_test['Median'] = median_guesses

# write prediction data to Excel
with pd.ExcelWriter('classification_data.xlsx', engine='openpyxl') as writer: 
    df_setosa.to_excel(writer, sheet_name='Setosa',index=False)
    df_versicolor.to_excel(writer, sheet_name='Versicolor',index=False)
    df_virginica.to_excel(writer, sheet_name='Virginica',index=False)
    df_test.to_excel(writer, sheet_name='Test Data & Results',index=False)

# Calculate accuracy for each prediction
means = []
mean_sd = []
medians = []
i = 0
for index, row in df_test.iterrows():
    if row['Species'] == mean_guesses[i]:
        means.append('true')
    else:
        means.append('false')
    if row['Species'] == mean_sd_guesses[i]:
        mean_sd.append('true')
    else:
        mean_sd.append('false')
    if row['Species'] == median_guesses[i]:
        medians.append('true')
    else:
        medians.append('false')
    i += 1

df_percent = pd.DataFrame(means)
df_percent.columns = ['means']
df_percent['mean_sd'] = mean_sd
df_percent['med'] = medians

# save accuracy calculations for each run to a text file
now = datetime.datetime.now()
run_time = (f'{now.strftime("%d/%m/%Y %H:%M:%S")}\n')
mean_accuracy = (f"Dist to Mean Accuracy = {round((df_percent.means.value_counts().true/45)*100, 2)}%\n")
mean_sd_accuracy = (f"Dist to Mean/Standard Deviation Accuracy = {round((df_percent.mean_sd.value_counts().true/45)*100, 2)}%\n")
median_accuracy = (f"Dist to Median Accuracy = {round((df_percent.med.value_counts().true/45)*100, 2)}%\n\n")

with open('accuracy.txt', 'a') as f:
    f.write(run_time)
    f.write(mean_accuracy)
    f.write(mean_sd_accuracy)
    f.write(median_accuracy)
