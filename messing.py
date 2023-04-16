# Download the data
# Author: Declan Fox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from sklearn.datasets import load_iris
iris = load_iris()
 
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])
 
