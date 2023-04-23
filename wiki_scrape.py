# scrape.py
# tests "scraping" the iris data set from Wikipedia
# Author: Declan Fox

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import csv
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

data_set = []
py_url = "http://en.wikipedia.org/wiki/Iris_flower_data_set"
py_page = requests.get (py_url)
py_soup = bs(py_page.text, 'html.parser')
py_table = py_soup.find ('table', {'class':'wikitable'})
py_rows = py_table.find_all ('tr')

df=pd.read_html(str(py_table))
# convert list to dataframe
df=pd.DataFrame(df[0])

with pd.ExcelWriter('iris.xlsx', engine='openpyxl', mode='a') as writer: 
    df.to_excel(writer, sheet_name='wiki',index=False)

