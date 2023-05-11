# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:14:14 2023

@author: aaron
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import cluster_tools as ct
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import errors as err


def read_df(fname):
    ''' Function to read a csv file in world bank format and clean it for
    clustering.
    Arguments:
        Name/file path of a csv file.
    Returns the dataframe and transposed dataframe.
    '''
    # reading from csv
    df = pd.read_csv(fname, skiprows=4)

    # dropping columns that are not needed
    df.drop(["Country Code", "Indicator Code", "Indicator Name"], axis=1,
            inplace=True)
    df
    # setting index
    df.set_index(['Country Name'], inplace=True)
    # naming column
    df.columns.name = "Year"
    # droping columns and rows with nan values
    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="any", thresh=50, inplace=True)
    # filling nan values
    df.fillna(0, inplace=True)

    # transposing
    df_t = df.transpose()
    # change dtype to int
    df_t.index = pd.to_numeric(df_t.index)
    # droping columns with nan values
    df_t.dropna(how='all', axis=1, inplace=True)
    return df, df_t


def cluster_plot(df, nc):
    ''' Function to plot clustering graph.
    Arguments:
        A dataframe.
        The number of clusters needed.
    Returns the cluster centres and array of labels.

    '''
    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=nc)
    # fit data
    kmeans.fit(df)
    # extracting labels
    labels = kmeans.labels_
    # adding labels to df
    df['Labels'] = labels
    # extracting cluster centres
    cen = kmeans.cluster_centers_

    # plotting
    plt.figure(figsize=(7, 7))

    # scatter plot
    colors = ['red', 'blue', 'green', 'orange', 'pink']
    labeling = ['Cluster '+str(i+1) for i in range(nc)]

    for i in range(nc):
        plt.scatter(df[df['Labels'] == i]['1965'],
                    df[df['Labels'] == i]['2015'],
                    c=colors[i], label=labeling[i])

    # plot cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80, label='Centres')

    plt.xlabel("1965")
    plt.ylabel("2015")
    plt.title(str(nc) + " Clusters")
    plt.legend()
    plt.show()
    return cen, labels


def logistic(x, a, b, c):
    f = c / (1 + np.exp(-a*(x-b)))
    return f


def poly(t, c0, c1, c2, c3):
    """ Computes a polynominal c0 + c1*t + c2*t^2 + c3*t^3
    Arguments:
        x value
        Constant
        Coefficient of x
        Coefficient of x^2
        Coefficient of x^3
    Returns f(x)
    """
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f


# read the data
df_ogc, df_ogc_t = read_df("API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5211704.csv")

# setting seaborn theme
sns.set_theme()

'''
# clustering

# plotting scatter matrix to find good cluster
years = ['1960', '1961', '1962', '1963', '1964', '1965',
         '2010', '2011', '2012', '2013', '2014', '2015']
pd.plotting.scatter_matrix(df_ogc[years], figsize=(12, 12))
plt.show()

# chosing best years for clustering
df_clstr = df_ogc[['1965', '2015']].copy()

# normalising and storing minimum and maximum
df_normalised, df_min, df_max = ct.scaler(df_clstr)

# calculating wcss for different no. of clusters
wcss = []
for i in range(1, 10):
    # set up kmeans and fit
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10)
    kmeans.fit(df_normalised)
    wcss.append(kmeans.inertia_)

# plotting to use elbow method
plt.figure(figsize=(14, 7))

plt.plot(range(1, 10), wcss, marker='o')

plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# plot for three clusters
cen3, labels3 = cluster_plot(df_normalised, 3)

# plot for four clusters
cen4, labels4 = cluster_plot(df_normalised, 4)

# backscale the normalised df
cen_bckscl = ct.backscale(cen4, df_min, df_max)

# backscaled cluster graph
plt.figure(figsize=(7, 7))

plt.scatter(df_ogc["1990"], df_ogc["2015"], c=labels4, cmap='Set1')

xcen = cen_bckscl[:, 0]
ycen = cen_bckscl[:, 1]

plt.scatter(xcen, ycen, 45, "k", marker="d")

plt.xlabel("1965")
plt.ylabel("2015")
plt.title("Backscaled Cluster Graph")
plt.show()
'''

# fitting

# data to fit
df_tofit = pd.DataFrame(df_ogc_t.iloc[:, 3].copy())

# plot data to be fitted
df_tofit.plot(title='Electricity Produced from Oil, Gas and Coal Sources',
              ylabel='Electricity Produced(% of Total)')
plt.show()

# fitting polynomial function
param, covar = opt.curve_fit(logistic, df_tofit.index, df_tofit.iloc[:, 0],
                             p0=(0.0, 1987.5, 100))

# add fit values to dataframe
df_tofit["fit"] = logistic(df_tofit.index, *param)

plt.figure()

plt.plot(df_tofit.index, df_tofit.iloc[:, 0], label="data")
plt.plot(df_tofit.index, df_tofit["fit"], label="fit")

# calculate sigma
sigmas = np.sqrt(np.diag(covar))

# calculate upper and lower limits
years = np.arange(1960, 2016)
lower, upper = err.err_ranges(years, logistic, param, sigmas)
# plot error ranges
plt.fill_between(years, lower, upper, alpha=0.5)

plt.legend()
plt.show()
