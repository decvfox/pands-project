# load_iris_data.py
# Author: Declan Fox
# Test writing all 3 datasets to 1 xl file


import csv
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import requests
from bs4 import BeautifulSoup as bs

# import from CSV file
FILENAME="iris_data.csv"
with open(FILENAME, "rt") as file:
    csvReader = csv.reader(file, delimiter = ',') 

# load to Pandas Dataframe
df1 = pd.read_csv('iris_data.csv')

# Download from API
iris = load_iris()
df2= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

# Scrape from website
py_url = "https://en.wikipedia.org/wiki/Iris_flower_data_set"
py_page = requests.get (py_url)
py_soup = bs(py_page.text, 'html.parser')
py_table = py_soup.find ('table', {'class':'wikitable'})
py_rows = py_table.find_all ('tr')

df3=pd.read_html(str(py_table))
# convert list to dataframe
df3=pd.DataFrame(df3[0])

# write to Excel
with pd.ExcelWriter('iris_dataset.xlsx', engine='openpyxl') as writer: 
    df1.to_excel(writer, sheet_name='CSV',index=False)
    df2.to_excel(writer, sheet_name='API',index=False)
    df3.to_excel(writer, sheet_name='Wiki',index=False)

