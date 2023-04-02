# Download the data
# Author: Declan Fox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from sklearn.datasets import load_iris
iris = load_iris()
 
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
 
X = df.iloc[0:150, [0, 1, 2, 3]].values

index = 1
for plant in X:
    print(f'{index}. = {plant[0]}, {plant[1]}, {plant[2]}, {plant[3]},')
    index += 1