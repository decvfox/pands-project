# variable_txt.py
# author Declan Fox
# tests writing variable summaries to text file.

FILENAME = "variables.txt"

Sepal_length = '''this will hold a summary of what sepal length is and the data it contains\n'''
Sepal_width = '''this will hold a summary of what sepal width is and the data it contains\n'''
Petal_length = '''this will hold a summary of what petal length is and the data it contains\n'''
Petal_width	= '''this will hold a summary of what petal width is and the data it contains\n'''
Species = '''this will hold a summary of what the species variable is and the data it contains\n'''

with open(FILENAME, 'wt') as f:
    f.write(Sepal_length)
    f.write(Sepal_width)
    f.write(Petal_length)
    f.write(Petal_width)
    f.write(Species)