# load_iris_data.py
# Author: Declan Fox

import csv
FILENAME="iris_data.csv"
with open(FILENAME, "rt") as file:
    csvReader = csv.reader(file, delimiter = ',') 
    index = 1
    for line in csvReader:
        if line:
            variety = line[4]
            sepal_length = line[0]
            sepal_width = line[1]
            petal_length = line[2]
            petal_width = line[3]
            print(f'{index}.= {sepal_length}, {sepal_width}, {sepal_length}, {sepal_width}, {variety}')
            index += 1