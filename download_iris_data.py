# Download the data
# Author: Declan Fox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import requests
from bs4 import BeautifulSoup as bs
import csv

iris = load_iris()
 
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
 
with pd.ExcelWriter('iris.xlsx', engine='openpyxl', mode='a') as writer: 
    df.to_excel(writer, sheet_name='sklearn',index=False)
    
