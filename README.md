# Programming and Scripting-Project

![](https://github.com/decvfox/pands-project/blob/main/Banner.png)

## Install
This project requires the following libraries:
|Library|Installation Instructions|
| -------|----- |
|CSV |https://pypi.org/project/python-csv/|
|Numpy|https://numpy.org/|https://numpy.org/install/|
|Pandas|https://pandas.pydata.org/docs/getting_started/install.html|
|Beautiful Soup 4|https://pypi.org/project/beautifulsoup4/|
|Matplotlib|https://matplotlib.org/stable/users/installing/index.html|
|Seaborn|https://seaborn.pydata.org/installing.html|
|Sklearn.datasets|https://scikit-learn.org/stable/install.html|

## Introduction
Fisher’s Iris Data Set is based on data collected by Edgar Anderson from three species of irises in the Gaspé Peninsula in Quebec, Canada. This Data set has been used countless times in statistics papers since they were first used by R. A. Fisher in 1936 to introduce the method of discriminant analysis. The dataset is made up of five variables, the first four of which are measurements of Sepal Length, Sepal Width, Petal Length and Petal Width, and a fifth variable which denotes which species of iris, Iris Setosa, Iris Versicolour, and Iris Virginca(Fox and Weisberg, 2011). There are 50 samples for each species for a total of 150 samples. Fishers aim was to classify the species based on the different measurements in the dataset and since the advent of machine learning this dataset has become a benchmark for testing algorithms.
**Sepals:** In most flowering plants sepals are usually the green leaf like structures that enclose the flower in the bud and open up to support the petals when it blooms. While Botanists disagree as to whether Irises have Sepals or not(‘What-should-we-know-about-the-famous-Iris-data.pdf’, 2013), for the purposes of this project, we will, as Anderson and Fisher have done, count the larger petal-like parts of the flower as sepals.


<p align="center">
<img src="https://github.com/decvfox/pands-project/blob/main/Sepal-Petal.png" width="320" height="235">
</p>

I will give a brief outline of my project below, for full descriptions of each step and code please see the attached Jupyter Notebook.

##### Dataset Collection
I will demonstrate three common ways of getting data for analysis. Manually downloading the data set and storing it as a CSV file on a local drive, using an API, Scraping  the data from a website.

##### Preparation
I will add headings, add or remove columns so I can save the datasets in the same format, this will also make the datasets easier to compare.

##### Dataset Comparison
I used the Pandas compare method to compare the 3 datasets. I used Wikipedia as one of my sources in the hopes that it wouldn't be as correct as the other two datasets but instead, I found more issues with UCI version.

##### Dataset Storage
I will store the datasets in 3 worksheets in the same Excel Workbook

##### Exploration and  Plotting
I will use matplotlib and seaborn to create plots that will make the data easier to visualise and understand.

##### Experiment and Predict
I have decided to see if I could create a method for classifying the samples myself rather than importing and using a library I might not fully understand.

### Conclusion
I found this project remarkably interesting, particularly using Pandas to store and manipulate data. I also enjoyed using Matplotlib and Seaborn to plot the data. I would have liked to have had a larger dataset to use these tools to their full potential. 
I use Excel a lot for work to store and filter data and I would have never considered using Python for these tasks before now, but I noted some pandas functions as part of my research for this project that would be really useful for these tasks. I have not used pandas before this course, and I found it really intuitive to use.
I used Matplotlib and Seaborn to help visualise the data, I haven’t used these modules before and, even though I found it a little harder to grasp than Pandas I am really happy with the results in particular the Seaborn heatmap.
I had a look at some of the scikit-learn modules such as k-means clustering, PCA and decision tree analysis but decided against using them as I didn’t completely understand the underlying maths and I would have ended up using code I didn’t fully understand.
Instead, I decided to create my own classifier by adding the distances from the centres of each set of variables to the values of the sample for classification. I tried the mean, the mean divided by the standard deviation and the median to find the centres.
This gave me results of between 85 and 100% accuracy. This accuracy seemed to be dependant on the amount of each species contained in the training data with less Setosa and more of the other two species giving more accuracy.
As this project is to be used as part of a presentation, I haven’t added error or exception handling as the script will be supervised at all times. 


### References:

Fox, J. and Weisberg, S. (2011) ‘Multivariate Linear Models in R’.

‘What-should-we-know-about-the-famous-Iris-data.pdf’ (2013). Available at: https://www.researchgate.net/profile/Marcin-Kozak-2/publication/237010807_What_should_we_know_about_the_famous_Iris_data/links/02e7e51be9229f3495000000/What-should-we-know-about-the-famous-Iris-data.pdf (Accessed: 16 April 2023).

Fisher, R.A., 1936. The use of multiple measurements in taxonomic problems. Annals of eugenics, 7(2), pp.179-188.

